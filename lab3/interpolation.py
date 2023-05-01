from numpy.polynomial.polynomial import Polynomial
import numpy as np


def get_step(a, b, n):
    return (b - a) / n


def uniform_points(a, b, n):
    h = (b - a) / n
    result = []
    for i in range(n + 1):
        result.append(a + h * i)
    return np.array(result)


def lagrange(x, y, domain=[-1, 1]):
    if x.shape != y.shape:
        raise ValueError("x and y don't have the same shape")
    x_count = x.shape[0]
    L = Polynomial([0], domain, domain)
    omega_cache = []
    omega_bottom_cache = np.eye(x_count)
    for i, curr_x in enumerate(x):
        omega_cache.append(Polynomial([-curr_x, 1], domain, domain))
        for j, curr_x_2 in enumerate(x):
            if i == j:
                continue
            omega_bottom_cache[i, j] = curr_x - curr_x_2
    for i, curr_x in enumerate(x):
        bot = np.prod(omega_bottom_cache[i, :])
        temp = y[i] / bot
        for j in range(x_count):
            if i == j:
                continue
            temp *= omega_cache[j]
        L += temp
    return L


def newton(x, y):
    x_count = x.shape[0]
    div_diffs = np.zeros((x_count, x_count))
    div_diffs[:, 0] = y
    for j in range(1, x_count):
        for i in range(x_count - j):
            div_diffs[i, j] = (div_diffs[i + 1, j - 1] - div_diffs[i, j - 1]) / (x[i + j] - x[i])
    N = div_diffs[0, 0]
    for i in range(1, x_count):
        temp = div_diffs[0, i]
        for j in range(i):
            temp *= Polynomial([-x[j], 1])
        N += temp
    return N


class CubicSpline:
    def __init__(self, x, funcs):
        self.ranges = x
        self.funcs = funcs

    def __call__(self, value):
        for i, r in enumerate(self.ranges):
            if r > value:
                return self.funcs[i - 1](value)
        # a = 0
        # b = self.ranges.shape[0]
        # while b - a <= 2:
        #     i_center = (b - a) // 2
        #     center = self.ranges[i_center]
        #     if value < center:
        #         b = i_center
        #     elif value >= center:
        #         a = i_center
        # return self.funcs[a](value)


def tridiagonal_gauss(A: np.ndarray, f: np.ndarray):
    c = -np.diag(A)
    a = np.diag(A, k=-1)
    b = np.diag(A, k=1)
    alpha_i = b[0] / c[0]
    f = f.copy().reshape((c.shape[0]))
    beta_i = -f[0] / c[0]
    # print(f"alpha_1: {alpha_i}, beta_1: {beta_i}")
    alpha = [alpha_i]
    beta = [beta_i]
    size = A.shape[0]
    x = np.zeros(size)
    for i in range(1, size - 1):
        z_i = c[i] - alpha_i * a[i - 1]
        # print(f"z_{i}: {z_i}")
        beta_i = (-f[i] + a[i - 1] * beta_i) / z_i
        alpha_i = b[i] / z_i
        # print(f"alpha_{i+1}: {alpha_i}, beta_{i+1}: {beta_i}")
        alpha.append(alpha_i)
        beta.append(beta_i)
    z_i = c[-1] - alpha_i * a[-1]
    # print(f"z_{size - 1}: {z_i}")
    x[-1] = (-f[-1] + a[-1] * beta_i) / z_i
    for i in range(size - 1)[::-1]:
        x[i] = x[i + 1] * alpha[i] + beta[i]
    return x


def cubic_spline(x, y, h):
    x_count = x.shape[0] - 1
    A_shape = (x_count - 1, x_count - 1)
    A = np.zeros(A_shape)
    A += np.diag([(2 * h) / 3] * (x_count - 1))
    ul_diags = [h / 6] * (x_count - 2)
    A += np.diag(ul_diags, -1)
    A += np.diag(ul_diags, 1)
    H_shape = (x_count - 1, x_count + 1)
    H = np.zeros(H_shape)
    # print(H[:, x_count - 1])
    H[:, :x_count - 1] += np.diag([1 / h] * (x_count - 1))
    H[:, 1:x_count] += np.diag([-2 / h] * (x_count - 1))
    H[:, 2:x_count + 1] += np.diag([1 / h] * (x_count - 1))
    f = H @ y.T
    m = tridiagonal_gauss(A, f)
    M = np.zeros(x_count + 1)
    M[1:-1] = m
    result = CubicSpline(np.zeros(x_count), [])
    result.ranges = x.copy()
    for i, curr_x in enumerate(x):
        if i == 0:
            continue
        a = y[i]
        d = (M[i] - M[i - 1]) / h
        b = (h / 2 * M[i] - h * h / 6 * d + (y[i] - y[i - 1]) / h)
        d_x = Polynomial([-curr_x, 1])
        s = a + b * d_x + M[i] / 2 * d_x ** 2 + d / 6 * d_x ** 3
        result.funcs.append(s)
    return result
