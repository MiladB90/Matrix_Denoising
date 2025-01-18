#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy
from numpy.random import Generator
from pandas import DataFrame
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from google.cloud import bigquery

# import from other files
from data_generators import *
from shrinkers import *

from EMS.manager import active_remote_engine, do_on_cluster, unroll_experiment, get_gbq_credentials
from dask.distributed import Client, LocalCluster
import coiled
import logging
import json

logging.basicConfig(level=logging.INFO)


def _df(c: list, l: list) -> DataFrame:
    d = dict(zip(c, l))
    return DataFrame(data=d, index=[0])

def list_encoder(l: list) -> str:
    return json.dumps(l)

def list_decoder(s: str) -> list:
    return json.loads(s)

    

def do_matrix_denoising(*, m: int, n: int, rank: int, signal_strengths: str, p: float, sigma: float,
                         ensemble: str, left_singvec_dist: str, right_singvec_dist: str,
                         solver_name: str, solver_parameters: str, tune_mode: str,
                         max_matrix_dim: int, max_rank: int, max_solver_params: int, mc_id: int) -> DataFrame:
    # unpack list inputs, e.g. signal strenghts
    ells = list_decoder(signal_strengths)
    solver_parameters_list = list_decoder(solver_parameters)
    
    # generate data
    # functions are defined in data_generators.py
    rng = np.random.default_rng(seed=seed(m=m, n=n, p=p, signal_strngths=ells, solver_parameters=solver_parameters_list,
                                          mc_id=mc_id))
    U, V, noise, noise_entry_std = make_data(m=m, n=n, rank=rank, p=p, sigma=sigma,
                                           ensemble=ensemble, left_singvec_dist=left_singvec_dist,
                                           right_singvec_dist=right_singvec_dist,
                                           rng=rng)
    
    # form the model  (eta(signal + scale * noise; lambda))    
    signal = U @ np.diag(ells) @ V.T
    noisy_observations = signal + noise
    
    # estimate the signal
    shrinker_name, shrinker_parameters = get_shrinker_name_and_parameters(p, solver_name, solver_parameters_list, tune_mode)
    def eta(x):
        return shrinker(x, shrinker_name, shrinker_parameters)

    Uhat, Shat, Vhat = np.linalg.svd(noisy_observations)
    Vhat = Vhat.T       # transpose vhat to have singular vectors as columns
    Shat = [eta(s) for s in Shat]       # apply the shrinkage
    estimator = Uhat @ np.diag(Shat) @ Vhat.T
    
    # take measurements
    df_out = take_measurements(U=U, V=V, signal=signal, rank=rank, noisy_observations=noisy_observations,
                               estimator=estimator,
                               max_rank=max_rank, max_matrix_dim=max_matrix_dim)

    # concatenate inputs + outputs
    # input
    c = 'm, n, rank, signal_strengths, p, sigma, noise_entry_std,' \
        ' ensemble, left_singvec_dist, right_singvec_dist,' \
        ' solver_name, solver_parameters, shrinker_name, shrinker_parameters, tune_mode,' \
        ' max_matrix_dim, max_rank, max_solver_params, mc_id'.split(', ')
    inputs = [m, n, rank, signal_strengths, p, sigma, noise_entry_std,
              ensemble, left_singvec_dist, right_singvec_dist,
              solver_name, solver_parameters, shrinker_name, list_encoder(shrinker_parameters), tune_mode,
              max_matrix_dim, max_rank, max_solver_params, mc_id]
    df_inputs = _df(c, inputs)
    
    # unpack signal strength, put it in a df with unified number of columns max_rank
    full_ells = [0] * max_rank
    full_ells[:rank] = ells
    c = [f'ell_{i}' for i in range(max_rank)]
    df_signal_strengths = _df(c, full_ells)

    # unpack solver params, put it in a df with unified number of columns max_rank
    full_shrinker_parameters = [None] * max_solver_params
    param_size = len(shrinker_parameters)
    full_shrinker_parameters[:param_size] = shrinker_parameters
    cols = [f'lambda_{i}' for i in range(max_solver_params)]
    df_shrinker_parameters = _df(cols, full_shrinker_parameters)

    # concat output
    df_inputs = pd.concat([df_inputs, df_signal_strengths, df_shrinker_parameters], axis=1)
    df = pd.concat([df_inputs, df_out], axis=1)
    
    return df


# measurements
def take_measurements(*, U: np.ndarray, V: np.ndarray, signal: np.ndarray, noisy_observations: np.ndarray, rank: int,
                        estimator: np.ndarray,
                        max_rank: int, max_matrix_dim: int) -> DataFrame:

    # to avoid svd did not converge first normalize the estimator, and use scipy svd
    factor =  np.linalg.norm(estimator)
    Uhat, Shat, Vhat = scipy.linalg.svd(estimator / factor, full_matrices=False)
    Shat *= factor
    # transpose Vhat to get vectors as columns
    Vhat = Vhat.T

    measures = {}
    # 1. left cos similarities (cos_l_{i},  i = 0, 1, ..., rank)
    for i in range(max_rank):
        name = f'cos_l_{i}'
        val = np.abs(np.dot(U[:, i], Uhat[:, i])) if i < rank else None
        measures[name] = val

    # 2. right cos similarities (cos_r_{i},  i = 0, 1, ..., rank)
    for i in range(max_rank):
        name = f'cos_r_{i}'
        val = np.abs(np.dot(V[:, i], Vhat[:, i])) if i < rank else None
        measures[name] = val

    # 3. spectrum of estimator (sv_{i}, i = 0, 1, ..., max_matrix_dim - 1)
    for i in range(max_matrix_dim):
        val = Shat[i] if i < len(Shat) else None
        name = f'sv_{i}'
        measures[name] = val

    # 4. MSE with signal (MSE_signal)
    name = 'MSE_signal'
    val = get_mse(signal, estimator)
    measures[name] = val

    # 5. MSE with full noisy observations (MSE_obs)
    name = 'MSE_obs'
    val = get_mse(noisy_observations, estimator)
    measures[name] = val

    # 6. true nuc norm of full noisy observation
    name = 'nuc_norm_full_noisy_obs'
    val = np.linalg.norm(noisy_observations, 'nuc')
    measures[name] = val

    # 7. estimated nuc norm
    name = 'nuc_norm_est'
    val = np.linalg.norm(estimator, 'nuc')
    measures[name] = val

    # 8. relative Frobenius norm of error
    name = 'relative_err_fro_norm_noisy_obs'
    err = noisy_observations - estimator
    val = np.linalg.norm(err, 'fro') / np.linalg.norm(noisy_observations, 'fro')
    measures[name] = val

    # 9. relative Frobenius norm of error
    name = 'relative_err_fro_norm_signal'
    err = signal - estimator
    val = np.linalg.norm(err, 'fro') / np.linalg.norm(signal, 'fro')
    measures[name] = val


    # make dataframe and return
    measures_df = DataFrame(measures, index=[0])
    return measures_df

# other functions
def get_mse(mat1: np.ndarray, mat2: np.ndarray) -> float:
    diff = mat1 - mat2
    mse = (diff ** 2).mean()
    return mse


def test_experiment() -> dict:
    # make sure dim, rank, and number of solver parameters are upperbounded by below number through the entire experiment
    max_matrix_dim = 500
    max_rank = 5
    max_solver_params = 2
    author = 'milad'
    exp = dict(table_name=f'{author}_md_0024',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=[]
               )
    mr = exp['multi_res']
    ell_values = [p for p in np.linspace(2, 100, 10)] + [1000]
    ranks = [1, 2, 3, 4, 5]
    ps = [round(p, 3) for p in np.linspace(0.01, 1, 10)]
    m = n = 500
    for ell in ell_values:
        for rank in ranks:
            for p in ps:
                d = {
                    'm': [m],
                    'n': [n],
                    'rank': [rank],
                    'signal_strengths': [list_encoder([round(ell, 3)] * rank)],
                    'p': [p],
                    'sigma': [1],
                    'tune_mode': ['no_shrink'],
                    'ensemble': ['gaussian_1_over_p_row_var'],
                    'left_singvec_dist': ['orthogonal'],
                    'right_singvec_dist': ['orthogonal'],
                    'solver_name': ['norm_nuc_pen'],
                    'solver_parameters': [list_encoder([round(p, 3)]) for p in np.linspace(0.1, 5 * np.sqrt(p), 10)],
                    'mc_id': [1],

                    # size unifying parameters
                    'max_matrix_dim': [max_matrix_dim],
                    'max_rank': [max_rank],
                    'max_solver_params': [max_solver_params]
                }
                mr.append(d)
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
    inds = [0, 1, -1, -2]
    # inds = []
    inds += list(np.random.randint(len(params), size=6, dtype=int))
    print(f'experiment has {len(params)} items. we test run on random sample {inds}.')
    t0 = time()
    df = DataFrame()
    for ind in inds:
        p = params[ind]
        print(f' ind = {ind}, passed params {p}\n')
        df = pd.concat([df, do_matrix_denoising(**p)], ignore_index=True)
    pd.set_option('display.max_columns', None)
    # print(df)
    print(df.shape)
    no_sv_columns = [col for col in df.columns if 'sv_' not in col]
    sv_columns = [col for col in df.columns if 'sv_'  in col]
    cols_to_show = no_sv_columns + sv_columns[:2] + sv_columns[-2:]
    print(df[cols_to_show])

    def get_run_time(start):
        from time import time
        d = time() - start
        return f'{round(d / 60, 2)} mins'
    print(f'run time for {len(inds)} runs: {get_run_time(t0)}')
    d = time() - t0
    est = round((d * len(params) / len(inds)) / 3600, 2)
    print(f'whole run time estimate for {len(params)} items on one core is {est} hours')


if __name__ == "__main__":
    do_local_experiment()
    # do_coiled_experiment()
    # do_test()
