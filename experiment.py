#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
from numpy.random import Generator
from cvxpy.atoms import normNuc, multiply, norm
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




def df_experiment(m: int, n: int, snr: float, p: float, mc: int, svv: np.array) -> DataFrame:
    c = ['m', 'n', 'snr', 'p', 'mc']
    d = [m, n, snr, p, mc]
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


# problem setup
# def nuc_norm_problem(Y, observed) -> tuple:
#     X = cp.Variable(Y.shape)
#     objective = cp.Minimize(normNuc(X))
#     Z = multiply(X - Y, observed)
#     constraints = [Z == 0]

#     prob = cp.Problem(objective, constraints)

#     prob.solve()

#     return X, prob


# measurements
# def vec_cos(v: np.array, vhat: np.array):
#     return np.abs(np.inner(v, vhat))


# def take_measurements_svv(Mhat, u, v, noise):
#     uhatm, svv, vhatmh = np.linalg.svd(Mhat, full_matrices=False)
#     cosL = vec_cos(u, uhatm[:, 0])
#     cosR = vec_cos(v, vhatmh[0, :])

#     # make noise_spectrum
#     m, n = Mhat.shape
#     noise_spectrum = np.linalg.svd(noise, compute_uv=False)

#     # extract non-zero svs:
#     r1 = sum(svv > 0.001)
#     r2 = sum(noise_spectrum > 0.001)
#     r = min(r1, r2)

#     regr = LinearRegression()
#     X = noise_spectrum[1:r].reshape(-1,1)
#     Y = svv[1:r].reshape(-1,1)

#     regr.fit(X, Y)
#     slope = regr.coef_[0, 0]
#     intercept = regr.intercept_[0]
#     r_squared = regr.score(X, Y)

#     return cosL, cosR, svv, slope, intercept, r_squared

def do_matrix_denoising(*, m: int, n: int, snr: float, p: float, mc: int, max_matrix_dim: int) -> DataFrame:
    rng = np.random.default_rng(seed=seed(m, n, snr, p, mc))

    u, v, M, noise, obs, entr_noise_std = make_data(m, n, p, rng)
    Y = snr * M + noise

    svv = np.svd(Y, compute_uv=False)
    # fixed the length of svv for all runs
    fullsvv = np.full([max_matrix_dim], np.nan)
    fullsvv[:len(svv)] = svv

    return df_experiment(m, n, snr, p, mc, svv)
    


def test_experiment() -> dict:
   
    exp = dict(table_name='test_sherlock',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=[{
                   'm': [100, 200, 300],
                   'n': [100],
                   'snr': [4],
                   'p': [round(0.3, 3)],
                   'mc': [round(p) for p in np.linspace(1, 10, 10)]
               }])

    # this makes 38k runs 
    # add max_matrix_dim for having unified output size
    mr = exp['multi_res']
    max_matrix_dim = 0
    for params in mr:
        max_matrix_dim = max([[max_matrix_dim], params['m'], params['n']])[0]
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
            do_on_cluster(exp, do_matrix_completion, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_completion, client, credentials=get_gbq_credentials())


def do_test():
    print(get_gbq_credentials())
    exp = test_experiment()
    import json
    j_exp = json.dumps(exp, indent=4)
    print(j_exp)
    params = unroll_experiment(exp)
    for p in params:
        df = do_matrix_completion(**p)
        print(df)
    pass
    # df = do_matrix_completion(m=100, n=100, snr=10., p=2./3., mc=20)
    # df = do_matrix_completion(m=12, n=8, snr=20., p=2./3., mc=20)
    # print(df)


if __name__ == "__main__":
    # do_local_experiment()
    # do_coiled_experiment()
    do_test()
