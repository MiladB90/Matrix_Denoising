#!/usr/bin/env python3
import numpy as np
import pandas as pd
from numpy.random import Generator
from pandas import DataFrame
from scipy import stats as st
from sklearn.linear_model import LinearRegression

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




def df_experiment(m: int, n: int, snr: float, snr2: float, p: float, noise_scale: float, soft_lvl: float, max_matrix_dim: int, mc: int,
                   cos_l: float, cos_r: float, cos_l2: float, cos_r2: float, svv: np.array) -> DataFrame:

    # input
    c = ['m', 'n', 'snr', 'snr2', 'p', 'noise_scale', 'soft_lvl', 'max_matrix_dim', 'mc']
    d = [m, n, snr, snr2, p, noise_scale, soft_lvl, max_matrix_dim, mc]

    # output
    c += ['cosL', 'cosR', 'cosL2', 'cosR2']
    d += [cos_l, cos_r, cos_l2, cos_r2]
    for i, sv in enumerate(svv):
        c.append(f'sv{i}')
        d.append(sv)
    return _df(c, d)   


def make_data(m: int, n: int, p: float, rng: Generator) -> tuple:
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u2 = rng.normal(size=m)
    v2 = rng.normal(size=n)

    for vec in [u, v, u2, v2]:
        vec /= np.linalg.norm(vec)

    entr_noise_std = 1 / np.sqrt(n) 
    noise = rng.normal(0, entr_noise_std, (m, n))
    observes = st.bernoulli.rvs(p, size=(m, n), random_state=rng)

    return u, v, u2, v2, noise, observes, entr_noise_std


# measurements
def vec_cos(v: np.array, vhat: np.array):
    return np.abs(np.inner(v, vhat))


def take_measurements_svv(Y, u, v, u2, v2, soft_lvl):
    uhatm, svv, vhatmh = np.linalg.svd(Y, full_matrices=False)
    cos_l = vec_cos(u, uhatm[:, 0])
    cos_r = vec_cos(v, vhatmh[0, :])
    cos_l2 = vec_cos(u2, uhatm[:, 1])
    cos_r2 = vec_cos(v2, vhatmh[1, :])

    svv_soft = np.array([max(0, svi - soft_lvl) for svi in svv])

    return cos_l, cos_r, cos_l2, cos_r2, svv_soft


def do_matrix_denoising(*, m: int, n: int, snr: float, snr2: float, p: float, noise_scale: float, soft_lvl: float,
                         max_matrix_dim: int, mc: int) -> DataFrame:
    
    rng = np.random.default_rng(seed=seed(m, n, snr, p, mc))
                            
    u, v, u2, v2, noise, observes, entr_noise_std = make_data(m, n, p, rng)
    Y = (snr * np.outer(u, v)) + (snr2 * np.outer(u2, v2)) + (noise_scale * noise)

    cos_l, cos_r, cos_l2, cos_r2, svv_soft = take_measurements_svv(Y=Y, u=u, v=v, u2=u2, v2=v2, soft_lvl=soft_lvl)
                        
    # fixed the length of svv for all runs
    fullsvv = np.full([max_matrix_dim], np.nan)
    fullsvv[:len(svv_soft)] = svv_soft

    return df_experiment(m=m, n=n, snr=snr, snr2=snr2, p=p, noise_scale=noise_scale, soft_lvl=soft_lvl, max_matrix_dim=max_matrix_dim, mc=mc,
                         cos_l=cos_l, cos_r=cos_r, cos_l2=cos_l2, cos_r2=cos_r2, svv=fullsvv)
    


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
    
def test_experiment() -> dict:
    # This file name need to be changed for newly obtained corresponding model
    tune_file_name = 'tune_milad_mc_0019.csv'
    
    exp = dict(table_name='milad_md_0005',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=dict_from_csv(tune_file_name, mc_range=(11, 100))
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
    # print(j_exp)
    params = unroll_experiment(exp)
    print(params[0])
    for ind in [0, 1, 1000, 2000, 10000, -2, -1] :
        p = params[ind]
        start = time()
        df = do_matrix_denoising(**p)
        print(p, '\n', df.iloc[:, :20], f'\n run time = {round(time() - start, 3)}')
        print(df['cosL, cosR, cosL2, cosR2'.split(', ') + [f'sv{i}' for i in range(10)]], '\n'*2)

    pass
    
    # print(exp['multi_res'][:10])
    # print(exp['multi_res'][-10:])


if __name__ == "__main__":
    do_local_experiment()
    # do_coiled_experiment()
    # do_test()
