from . import helper
import numpy as np

def get_random_matrix(size):
    while True:
        A = (np.random.randint(1, 10, (size, size))).astype(np.intp)
        if np.linalg.det(A) > 10e-8:
            break
    return A

def get_random_system(size, x = None):
    A = get_random_matrix(size)
    if x is None:
        x = np.ones((size, 1), dtype=int) * 2
    else:
        x = np.array(x).reshape(size, 1)
    b = np.matmul(A, x)
    return A, b

def get_random_tridiag_system(size, x = None):
    A = get_random_matrix(size)
    if x is None:
        x = np.ones((size, 1), dtype=int) * 2
    else:
        x = np.array(x).reshape(size, 1)
    helper.make_matrix_diagdom(A)
    A_tridiag = np.diag(np.diag(A, k=-1), k=-1) + np.diag(np.diag(A)) + np.diag(np.diag(A, k=1), k=1)
    b = np.matmul(A_tridiag, x)
    return A_tridiag, b

def get_random_diagdom_system(size, x = None):
    A = get_random_matrix(size)
    if x is None:
        x = np.ones((size, 1), dtype=int) * 2
    else:
        x = np.array(x).reshape(size, 1)
    helper.make_matrix_diagdom(A)
    b = np.matmul(A, x)
    return A, b

def get_hilbert_system(size, x = None):
    if x is None:
        x = np.ones((size, 1), dtype=int) * 2
    else:
        x = np.array(x).reshape(size, 1)
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            A[i, j] = 1 / (i + j + 1)
    b = np.matmul(A, x)
    return A, b
