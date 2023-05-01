import PySimpleGUI as sg
# from polynomial import Polynomial, ABCPolyBase
import numpy as np
import matplotlib.pyplot as plt
import interpolation

def function(x):
    return np.sin(x)

if __name__ == "__main__":
    a = -10
    b = 10
    n = 15

    x = interpolation.uniform_points(a, b, n)
    L = interpolation.lagrange(x, function(x))
    N = interpolation.newton(x, function(x))
    S = interpolation.cubic_spline(x, function(x), interpolation.get_step(a, b, n))

    print(f"Lagrange: {L}")
    print(f"Newton: {N}")

    step = 0.1
    graph_padding = 1
    D = np.arange(a - graph_padding, b + graph_padding + step, step)
    spline_D = np.arange(a, b + step, step)

    spline_result = []
    for d in spline_D:
        spline_result.append(S(d))

    plt.scatter(x, function(x))
    original, = plt.plot(D, function(D))
    lagrange, = plt.plot(D, L(D))
    newton, = plt.plot(D, N(D))
    spline, = plt.plot(spline_D, spline_result)

    original.set_label("Original")
    lagrange.set_label("Lagrange")
    newton.set_label("Newton")
    spline.set_label("Spline")

    plt.grid(visible=True)
    plt.legend()
    plt.show()
