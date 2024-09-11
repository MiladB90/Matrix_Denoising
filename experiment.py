#!/usr/bin/env python3
import numpy as np
import pandas as pd
from numpy.random import Generator
from pandas import DataFrame
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from google.cloud import bigquery

from EMS.manager import active_remote_engine, do_on_cluster, unroll_experiment, get_gbq_credentials
from dask.distributed import Client, LocalCluster
import coiled
import logging

logging.basicConfig(level=logging.INFO)


def seed(m: int, n: int, snr: float, p: float, mc: int) -> int:
    return round(1 + m * 1000 + n * 1000 + round(snr * 1000) + round(p * 1000) + mc * 100000)


def _df(c: list, l: list) -> DataFrame:
    d = dict(zip(c, l))
    return DataFrame(data=d, index=[0])




def df_experiment(m: int, n: int, snr: float, p: float, noise_scale: float, soft_lvl: float, max_matrix_dim: int, mc: int,
                   cosL: float, cosR: float, svv: np.array) -> DataFrame:

    # input
    c = ['m', 'n', 'snr', 'p', 'noise_scale', 'soft_lvl', 'max_matrix_dim', 'mc']
    d = [m, n, snr, p, noise_scale, soft_lvl, max_matrix_dim, mc]

    # output
    c +=  ['cosL', 'cosR']
    d += [cosL, cosR]
    for i, sv in enumerate(svv):
        c.append(f'sv{i}')
        d.append(sv)
    return _df(c, d)   


def make_data(m: int, n: int, p: float, rng: Generator) -> tuple:
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    M = np.outer(u, v)
    entr_noise_std = 1 / np.sqrt(n) 
    noise = rng.normal(0, entr_noise_std, (m, n))
    observes = st.bernoulli.rvs(p, size=(m, n), random_state=rng)

    return u, v, M, noise, observes, entr_noise_std   



# measurements
def vec_cos(v: np.array, vhat: np.array):
    return np.abs(np.inner(v, vhat))


def take_measurements_svv(Y, u, v, soft_lvl):
    uhatm, svv, vhatmh = np.linalg.svd(Y, full_matrices=False)
    cosL = vec_cos(u, uhatm[:, 0])
    cosR = vec_cos(v, vhatmh[0, :])
    svv_soft = np.array([max(0, svi - soft_lvl) for svi in svv])

    return cosL, cosR, svv_soft


def do_matrix_denoising(*, m: int, n: int, snr: float, p: float, noise_scale: float, soft_lvl: float, 
                         max_matrix_dim: int, mc: int) -> DataFrame:
    
    rng = np.random.default_rng(seed=seed(m, n, snr, p, mc))
                            
    u, v, M, noise, obs, entr_noise_std = make_data(m, n, p, rng)
    Y = (snr * M) + (noise_scale * noise)                        

    cosL, cosR, svv_soft = take_measurements_svv(Y=Y, u=u, v=v, soft_lvl=soft_lvl) 
                        
    # fixed the length of svv for all runs
    fullsvv = np.full([max_matrix_dim], np.nan)
    fullsvv[:len(svv_soft)] = svv_soft

    return df_experiment(m=m, n=n, snr=snr, p=p, noise_scale=noise_scale, soft_lvl=soft_lvl, max_matrix_dim=max_matrix_dim, mc=mc,
                         cosL=cosL, cosR=cosR, svv=fullsvv)
    


def dict_from_csv(add: str, rename_cols=None, drop_cols=None, mc_range=(11, 20)) -> list:
  
  df = pd.read_csv(add, index_col=0)
  
  # below columns will be renamed
  if not rename_cols:
    rename_cols = {'nsspecfit_slope': 'noise_scale', 'nsspecfit_intercept':'soft_lvl'}
  # below columns will be drop
  if not drop_cols:
    drop_cols = ['nsspecfit_r2']
    
  df = df.drop(columns=drop_cols)
  df = df.rename(columns=rename_cols)
  
  # make positive soft thresholding level
  df['soft_lvl'] = np.abs(df['soft_lvl'])

  unique_dic = df.to_dict('records')

  multi_res = []
  for d in unique_dic: 
    # putting single values in a list
    for key in d.keys():
      d[key] = [d[key]]
        
    d['mc'] = [round(p) for p in np.arange(mc_range[0], mc_range[1] + 1, 1)]
    multi_res += [d]
  return multi_res

def make_tune_data(sensing_model_table_name):
    # load table_name
    project_id = 'hs-deep-lab-donoho'
    qr = f'SELECT * FROM  `EMS.{sensing_model_table_name}` WHERE mc <= 10'
    client = bigquery.Client(project=project_id, credentials=get_gbq_credentials())
    df_tune = client.query(qr).to_dataframe()
    df_tune = df_tune.dropna()

    # group mean
    all_cols = 'm, n, snr, snr2, p'.split(', ')
    grid_cols = [col for col in all_cols if col in list(df_tune.columns)]
    gdf = df_tune.groupby(grid_cols)
    df_emp_param = gdf.mean().reset_index()

    final_cols = grid_cols + ['nsspecfit_slope', 'nsspecfit_intercept', 'nsspecfit_r2']
    df_emp_param = df_emp_param[final_cols]

    # save data
    add = f'tune_{sensing_model_table_name}.csv'
    df_emp_param.to_csv(add, float_format='%.6f')

    return df_emp_param

def test_experiment() -> dict:
   
    # below two lines need to modify. it will run denoising model on same grid of hyper-parameters as in sensing_model
    # given.  Monte Carlo numbers would be chosen in mc_range.  since we used mc = 1, ..., 10 for tuning, it's better to
    # use mc >= 11 for making test data.
    sensing_model_table_name = 'milad_mc_0027'
    mc_range = (11, 20)


    make_tune_data(sensing_model_table_name)
    exp = dict(table_name='milad_md_0008',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=dict_from_csv(f'tune_{sensing_model_table_name}.csv', mc_range=mc_range)
              )

    
    # add max_matrix_dim for having unified output size
    mr = exp['multi_res']
    max_matrix_dim = 0
    for params in mr:
        paramlist =[max_matrix_dim]
        paramlist.extend(params['m'])
        paramlist.extend(params['n'])
        max_matrix_dim = max(paramlist)
    for params in mr:
        params['max_matrix_dim'] = [int(max_matrix_dim)]
    return exp


def do_coiled_experiment():
    exp = test_experiment()
    # logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/matrix_completion'
    # logging.info('Deleting environment.')
    # coiled.delete_software_environment(software_environment)
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment, n_workers=80) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_denoising, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_denoising, client, credentials=get_gbq_credentials())


def do_test():
    from time import time
    # print(get_gbq_credentials())
    exp = test_experiment()
    import json
    j_exp = json.dumps(exp, indent=4)
    print(j_exp)
    params = unroll_experiment(exp)
    inds = [0, 1, -1, -2]
    # inds = []
    inds += list(np.random.randint(len(params), size=6, dtype=int))
    print(f'experiment has {len(params)} items. we test run on random sample {inds}.')
    t0 = time()
    for ind in inds:
        p = params[ind]
        df = do_matrix_denoising(**p)
        print(f'ind = {ind}')
        cols = df.columns
        print(df[cols[:20]])


    def get_run_time(start):
        from time import time
        d = time() - t0
        return f'{round(d / 60, 2)} mins'

    print(f'run time for {len(inds)} runs: {get_run_time(t0)}')
    d = time() - t0
    est = round((d * len(params) / len(inds)) / 3600, 2)
    print(f'whole run time estimate on one core is {est} hours')


if __name__ == "__main__":
    do_local_experiment()
    # do_coiled_experiment()
    # do_test()
