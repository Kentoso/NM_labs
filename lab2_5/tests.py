import numpy as np
import pytest
from webgraph import WebGraph
from numpy.random import default_rng
import networkx as nx
import jacobi
import generator

ITERATIONS = 100

rng = default_rng()

@pytest.mark.parametrize('execution_number', range(ITERATIONS))
def test_pagerank(execution_number):
    tolerance = 10e-10
    epsilon = 10e-9

    alpha = 0.85
    print()

    g = WebGraph.generate(rng, 15)

    print(g.adj_matrix)

    result = g.pagerank(alpha, tolerance)
    print("My PageRank:", result)

    rows, cols = np.where(g.adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx_result = nx.pagerank(gr, alpha, tol=tolerance)
    nx_result = np.array([nx_result[p] for p in sorted(nx_result)])
    print("NetworkX PageRank:", nx_result)
    dif = np.linalg.norm(nx_result - result)
    print("Difference:", dif)
    assert dif < epsilon

@pytest.mark.parametrize('execution_number', range(100))
def test_jacobi(execution_number):
    print()
    # A = np.array([
    #     [2, -1, 0, 0],
    #     [-1, 2, -1, 0],
    #     [0, -1, 2, -1],
    #     [0, 0, -1, 2]
    # ])
    epsilon = 10e-5
    print("Start A:")
    A = generator.generate_random_symmetric(rng, 4)
    print(A)
    print("\n******************\n")
    vals, vects = jacobi.jacobi(A, epsilon)
    print("Jacobi Results: ")
    print(vals)
    print(vects)
    print("linalg.eigh Results:")
    test = np.linalg.eigh(A)
    print(test[0])
    print(test[1])
    assert np.allclose(vals, test[0])