import numpy as np

def make_matrix_diagdom(A):
    size = A.shape[0]
    for i in range(size):
        diag = np.abs(A[i, i])
        col_sum = np.sum(np.abs(A[i, :]))
        sum_of_other = col_sum - diag
        A[i, i] = sum_of_other * 2

def is_diagonally_dominant(A: np.ndarray):
    size = A.shape[0]
    all_geq = True
    one_greater = False
    for i in range(size):
        diag = np.abs(A[i, i])
        col_sum = np.sum(np.abs(A[i, :]))
        sum_of_other = col_sum - diag
        all_geq = all_geq and (diag >= sum_of_other)
        one_greater = one_greater or (diag > sum_of_other)
    return all_geq and one_greater

def print_system(A, b):
    print(np.hstack((A, b)))