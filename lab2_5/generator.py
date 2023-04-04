import numpy as np
import numpy.random


def generate_matrix_from_eigen(values: list[float], vectors: np.array):
    if (len(values) != vectors.shape[0]):
        return
    M = np.diag(np.array(values))
    S_inv = np.linalg.inv(vectors)
    return vectors @ M @ S_inv

def generate_random_symmetric(rng: numpy.random.Generator, n: int):
    a = np.triu(rng.integers(1, 25, (n, n)))
    return a + a.T - np.diag(np.diag(a))
