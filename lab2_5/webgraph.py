import numpy as np

class WebGraph:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix.copy()

    def pagerank(self, alpha = 0.85, epsilon = 10e-5):
        A = self.adj_matrix.copy().T
        n = A.shape[0]
        for i in range(n):
            col = A[:, i]
            A[:, i] = col / np.count_nonzero(col)
        M = np.ones(A.shape, dtype=float) / n
        B = alpha * A + (1 - alpha) * M
        x_prev = np.zeros((n, 1))
        x_k = np.ones((n, 1), dtype=float) / n
        while np.linalg.norm(x_k - x_prev) > epsilon:
            x_prev = x_k
            x_k = B @ x_k
        return x_k.reshape((-1))

    @staticmethod
    def generate(rng: np.random.Generator, n):
        adj = rng.integers(0, 2, (n, n)).astype(float)
        adj = adj - np.diag(np.diag(adj))
        zero_rows = np.where(~adj.any(axis=1))[0]
        for i in zero_rows:
            adj[i, i - 1] = 1
        zero_cols = np.where(~adj.any(axis=0))[0]
        for i in zero_cols:
            adj[i - 1, i] = 1
        return WebGraph(adj)
