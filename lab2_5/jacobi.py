import numpy as np

def jacobi(A: np.array, epsilon: float, max_iters=10000, non_converge_iters=30000):
    A = A.copy()
    AA = A * A
    t = (AA - np.diag(np.diag(AA))).sum(axis=None)
    n = A.shape[0]
    U_list = []
    i = 0
    while t > epsilon and i < max_iters and i < non_converge_iters:
        A_abs = np.triu(np.abs(A), 1)
        index = np.argmax(A_abs)
        i_k, j_k = np.unravel_index(index, A_abs.shape)
        a_ij = A[i_k, j_k]
        a_ii = A[i_k, i_k]
        a_jj = A[j_k, j_k]
        if a_ii == a_jj:
            phi = np.sign(a_ij) * np.pi / 4
        else:
            phi = np.arctan((2 * a_ij)/(a_ii - a_jj)) / 2
        U = np.eye(n)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        U[i_k, i_k] = cos_phi
        U[i_k, j_k] = -sin_phi
        U[j_k, i_k] = sin_phi
        U[j_k, j_k] = cos_phi
        U_list.append(U)
        print(f'////////////////////////{i}')
        print("phi", phi)
        print("sin", sin_phi, "cos", cos_phi)
        print("i", i_k, "j", j_k)
        print("U:")
        print(U)
        A = U.T @ A @ U
        A[np.abs(A) <= 10e-15] = 0
        print("A: ")
        print(A)
        i += 1
        AA = A * A
        t = (AA - np.diag(np.diag(AA))).sum(axis=None)
    if i >= non_converge_iters:
        print("Jacobi did not converge")
        return None
    values = np.diag(A)
    vectors = np.eye(n)
    for i in range(0, len(U_list)):
        vectors = vectors @ U_list[i]
    sort_ind = np.argsort(values)
    values = values[sort_ind]
    vectors[:, np.array(range(n))] = vectors[:, sort_ind]
    return values, vectors
