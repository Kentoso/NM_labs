import numpy as np
import pytest
from . import generator, helper, methods

if __name__ == "__main__":
    current_system = generator.get_random_system(3)
    print("SYSTEM: ")
    # A = np.array([[6, 5, 7], [8, 9, 1], [8, 9, 5]])
    # b = np.array([36, 36, 44]).reshape((-1, 1))
    A = np.array([[11, 2, 0], [1, 7, 4], [0, 4, 6]])
    b = np.array([13, 12, 10]).reshape((-1, 1))
    print(A)
    # print_system(*current_system)
    # gauss_with_main_element(current_system[0], current_system[1])
    # print_system(A, b)
    # gauss_with_main_element(A, b)
    print(helper.is_diagonally_dominant(A))
    # print_system(*get_random_tridiag_system(10))

    # Atd, btd = get_random_tridiag_system(3)
    Atd = np.array([[24, 5, 0], [2, 20, 8], [0, 3, 16]])
    btd = np.array([29, 30, 19]).reshape((-1, 1))
    helper.print_system(Atd, btd)
    print(methods.tridiagonal_gauss(Atd, btd))