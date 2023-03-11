import pytest
import numpy as np
from . import helper, generator, methods

SYSTEM_SIZE = 3
ITERATIONS = 100

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_gauss(execution_number):
    needed_x = [1] * SYSTEM_SIZE
    # needed_x = None
    current_system = generator.get_random_system(SYSTEM_SIZE, needed_x)
    print("")
    helper.print_system(*current_system)
    res = methods.gauss_with_main_element(current_system[0], current_system[1])
    assert np.allclose(res, np.array(needed_x))

@pytest.mark.parametrize('execution_number', range(1))
def test_gauss_hilbert(execution_number):
    needed_x = [1] * SYSTEM_SIZE
    current_system = generator.get_hilbert_system(SYSTEM_SIZE, needed_x)
    print("")
    helper.print_system(*current_system)
    res = methods.gauss_with_main_element(current_system[0], current_system[1])
    print(np.linalg.norm(np.array(needed_x) - res))
    assert np.allclose(res, np.array(needed_x), atol=10e-4)

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_tridiag_gauss(execution_number):
    needed_x = [1] * SYSTEM_SIZE
    current_system = generator.get_random_tridiag_system(SYSTEM_SIZE, needed_x)
    print("")
    helper.print_system(*current_system)
    res = methods.tridiagonal_gauss(current_system[0], current_system[1])
    print(res)
    assert np.allclose(res, np.array(needed_x))

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_diagdom_matrix_gen(execution_number):
    curr_system = generator.get_random_diagdom_system(SYSTEM_SIZE)
    print("")
    helper.print_system(*curr_system)
    assert helper.is_diagonally_dominant(curr_system[0])

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_jacobi(execution_number):
    needed_x = [1] * SYSTEM_SIZE
    epsilon = 10e-5
    current_system = generator.get_random_diagdom_system(SYSTEM_SIZE, x=needed_x)
    print("")
    helper.print_system(*current_system)
    res = methods.jacobi(current_system[0], current_system[1], epsilon)
    assert np.allclose(res, np.array(needed_x), atol=epsilon, rtol=0)

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_seidel(execution_number):
    needed_x = [1] * SYSTEM_SIZE
    epsilon = 10e-5
    current_system = generator.get_random_diagdom_system(SYSTEM_SIZE, x=needed_x)
    print("")
    helper.print_system(*current_system)
    res = methods.seidel(current_system[0], current_system[1], epsilon)
    assert np.allclose(res, np.array(needed_x), atol=epsilon, rtol=0)

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_random_matrix_cond(execution_number):
    A, _ = generator.get_random_system(SYSTEM_SIZE)
    print("")
    print(np.linalg.cond(A, 1))
    assert True

@pytest.mark.parametrize('execution_number', range(1))
def test_hilbert_matrix_cond(execution_number):
    A, _ = generator.get_random_system(SYSTEM_SIZE)
    A, _ = generator.get_hilbert_system(SYSTEM_SIZE)
    print("")
    print(np.linalg.cond(A, 1))
    assert True