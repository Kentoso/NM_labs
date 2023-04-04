import numpy as np
from webgraph import WebGraph
import networkx as nx
from numpy.random import default_rng
import matplotlib.pyplot as plt
import jacobi

if __name__ == '__main__':
    rng = default_rng()

    A = np.array([
        [2, -1, 0, 0],
        [-1, 2, -1, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 2]
    ])
    # A = np.array([
    #     [2, -1, 0],
    #     [-1, 2, -1],
    #     [0, -1, 2]
    # ])
    # A = np.array([
    #     [4, -30, 60, -35],
    #     [-30, 300, -675, 420],
    #     [60, -675, 1620, -1050],
    #     [-35, 420, -1050, 700]
    # ])
    jacobi.jacobi(A, 0.5)
    test = np.linalg.eigh(A)
    print(test[0])
    print(test[1])
    # 0.38197022
    b = np.array([-0.37092287, -0.60201015, -0.60201015, -0.37092287])
    print(A @ b.T)
    print(b * 0.38197022)
    # g = WebGraph.generate(rng, 30)
    #
    # rows, cols = np.where(g.adj_matrix == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    # gr = nx.DiGraph()
    # gr.add_edges_from(edges)

    # nx.draw(gr)
    # plt.show()

    # print(nx.nx_agraph.to_agraph(gr))

    # test_adj_m = np.array([
    #     [0, 0, 0, 1],
    #     [1, 0, 1, 1],
    #     [1, 0, 0, 1],
    #     [1, 1, 0, 0]
    # ]).astype(float)
    # g = Graph(test_adj_m)
    # print(g.pagerank())
    #
    # rows, cols = np.where(test_adj_m == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    # gr = nx.DiGraph()
    # gr.add_edges_from(edges)
    # print(nx.pagerank(gr))

    # test_adj_m = np.array([
    #     [0, 1, 1, 0],
    #     [0, 0, 0, 1],
    #     [1, 0, 0, 1],
    #     [0, 1, 0, 0]
    # ]).astype(float)
    # g = Graph(test_adj_m)
    # g.pagerank()
    #
    # rng = default_rng()
    # Graph.generate(rng, 2)
    #
    # rows, cols = np.where(test_adj_m == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    # gr = nx.DiGraph()
    # gr.add_edges_from(edges)
    # print(nx.pagerank(gr))