import numpy as np
from numpy.random import Generator

def seed(m: int, n: int, signal_strngths: list, p: float, solver_parameters: list,  mc_id: int) -> int:
    def list_to_num(l: list):
        coef = 1
        ans = 0
        for n in l:
            coef *= 10
            ans += coef * n
        return round(ans)
    return round(1 + m * 1000 + n * 1000 + list_to_num(signal_strngths) + round(p * 1000) + list_to_num(solver_parameters) + mc_id * 100000)


def make_data(m: int, n: int, rank: int, p: float, sigma: float,
              ensemble: str, left_singvec_dist: str, right_singvec_dist: str, rng: Generator):

    # singular vectors
    U = get_singular_vectors(m, rank, left_singvec_dist, rng)
    V = get_singular_vectors(n, rank, right_singvec_dist, rng)


    # noise
    if ensemble == 'gaussian_1_over_p_row_var':
        noise_entry_std = get_noise_entry_std(p=p, n=n, ensemble=ensemble)
        noise = rng.normal(0, noise_entry_std, (m, n))



    return U, V, noise, noise_entry_std

# noise scalar
def get_noise_entry_std(p, n, ensemble):
    if ensemble == 'gaussian_1_over_p_row_var':
        std = 1 / np.sqrt(p * n)

    return std


def gaussian_with_normalized_columns(row_size, col_size, rng):
    mat = rng.normal(size=(row_size, col_size))
    mat /= np.linalg.norm(mat, axis=0, keepdims=True)
    return mat

def get_singular_vectors(rows, cols, dist, rng):
    if dist == 'gaussian':
        singular_vectors = gaussian_with_normalized_columns(rows, cols, rng)
    elif dist == 'orthogonal':
        mat = rng.normal(size=(rows, cols))
        singular_vectors = gram_schmidt(mat)

    return singular_vectors

def gram_schmidt(A):
    """
    Orthonormalizes the columns of a matrix A using the Gram-Schmidt process.
    Args:
        A (np.ndarray): A matrix whose columns are to be orthonormalized.
    Returns:
        np.ndarray: The orthonormalized matrix.
    """

    Q = np.zeros_like(A)
    for i in range(A.shape[1]):
        v = A[:, i]
        for j in range(i):
            v -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        Q[:, i] = v / np.linalg.norm(v)
    return Q