import numpy as np
from . import helper

def gauss_with_main_element(A: np.array, b: np.array):
    print("Gauss")
    A_k = A.copy()
    b_c = b.copy()
    print()
    k = -1
    size = A.shape[0]
    p = 0
    while k + 1 < size:
        k += 1
        col_below_curr = A_k[k:, k]
        a_max_l = np.argmax(np.abs(col_below_curr), axis=0)
        if a_max_l == 0:
            p += 1
        P_kl = np.eye(size, dtype=int)
        P_kl[[k, a_max_l + k], :] = P_kl[[a_max_l + k, k], :]
        P_kl_A = np.matmul(P_kl, A_k)
        M_k = np.eye(size, dtype=float)
        if P_kl_A[k, k] == 0:
            raise RuntimeError("System was linearly dependent")
        M_k[k, k] = 1 / P_kl_A[k, k]
        M_k[k+1:, k] = - P_kl_A[k+1:, k] / P_kl_A[k, k]
        A_k = np.matmul(M_k, P_kl_A)
        b_c = np.matmul(M_k, np.matmul(P_kl, b_c))
        print("//////")
        helper.print_system(A_k, b_c)
    x = np.zeros(size)
    for i in range(0, size)[::-1]:
        x[i] = b_c[i] - np.sum(A_k[i, i+1:] * x[i+1:])
    print("Result: " + str(x))
    return x

def tridiagonal_gauss(A: np.ndarray, f: np.ndarray):
    c = -np.diag(A)
    a = np.diag(A, k=-1)
    b = np.diag(A, k=1)
    alpha_i = b[0] / c[0]
    f = f.copy().reshape((c.shape[0]))
    beta_i = -f[0] / c[0]
    print(f"alpha_1: {alpha_i}, beta_1: {beta_i}")
    alpha = [alpha_i]
    beta = [beta_i]
    size = A.shape[0]
    x = np.zeros(size)
    for i in range(1, size - 1):
        z_i = c[i] - alpha_i * a[i - 1]
        print(f"z_{i}: {z_i}")
        beta_i = (-f[i] + a[i - 1] * beta_i) / z_i
        alpha_i = b[i] / z_i
        print(f"alpha_{i+1}: {alpha_i}, beta_{i+1}: {beta_i}")
        alpha.append(alpha_i)
        beta.append(beta_i)
    z_i = c[-1] - alpha_i * a[-1]
    print(f"z_{size - 1}: {z_i}")
    x[-1] = (-f[-1] + a[-1] * beta_i) / z_i
    for i in range(size - 1)[::-1]:
        x[i] = x[i + 1] * alpha[i] + beta[i]
    return x

def jacobi(A: np.ndarray, b: np.ndarray, epsilon: float = 10e-5):
    D = np.diag(A)
    D_inv = np.diag(1 / D)
    size = A.shape[0]
    A_sum = A - np.diag(D)
    x_prev = np.zeros((size, 1))
    x_k = (- D_inv @ A_sum @ x_prev + D_inv @ b)
    i = 0
    while np.linalg.norm(x_k - x_prev) > epsilon:
        i += 1
        x_prev = x_k
        x_k = (- D_inv @ A_sum @ x_k + D_inv @ b)
        # x_prev = temp
        print(i)
        print(x_k)
    print("result:")
    print(x_k)
    return x_k.reshape(size)

def seidel(A: np.ndarray, b: np.ndarray, epsilon: float = 10e-5):
    A_1 = np.triu(A, k=1)
    A_2_D_inv = np.linalg.inv(np.tril(A))
    # D = np.diag(np.diag(A))
    # print(A_1)
    # print(A_2_D)
    size = A.shape[0]
    x_prev = np.zeros((size, 1))
    x_k = - A_2_D_inv @ A_1 @ x_prev + A_2_D_inv @ b
    i = 0
    while np.linalg.norm(x_k - x_prev) > epsilon:
        i += 1
        x_prev = x_k
        x_k = - A_2_D_inv @ A_1 @ x_k + A_2_D_inv @ b
        print(i)
        print(x_k)
    print("result:")
    print(x_k)
    return x_k.reshape(size)