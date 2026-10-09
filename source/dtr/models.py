"""From-scratch logistic regression (numpy only) - the interpretable baseline."""
from __future__ import annotations
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

class LogisticRegressionScratch:
    """Batch gradient descent on binary cross-entropy with L2, features standardised internally."""

    def __init__(self, lr=0.1, n_iter=2000, l2=1e-3, seed=0):
        self.lr, self.n_iter, self.l2, self.seed = lr, n_iter, l2, seed
        self.w = self.b = self.mu = self.sd = None
        self.loss_history: list[float] = []

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        self.w, self.b = rng.normal(0, 0.01, Z.shape[1]), 0.0
        n, eps = len(y), 1e-12
        self.loss_history = []
        for _ in range(self.n_iter):
            p = sigmoid(Z @ self.w + self.b)
            self.loss_history.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                                           + 0.5 * self.l2 * np.sum(self.w ** 2)))
            err = p - y
            self.w -= self.lr * (Z.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(((np.asarray(X, float) - self.mu) / self.sd) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)