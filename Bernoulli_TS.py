
from __future__ import annotations
import numpy as np

class Bernoulli_TS(object):

    def __init__(self, alpha:float, beta:float, K:int) -> None:
        if alpha < 0:
            raise ValueError("alpha should be > 0")
        if beta < 0:
            raise ValueError("beta should be > 0")

        self.alpha = alpha
        self.beta = beta

        self.S = [0 for _ in range(K)]
        self.F = [0 for _ in range(K)]
        self.K = K

    def get_action(self):
        # Draws thetas according to a Beta with pars
        # n_success_i + alpha , n_failures_i + beta
        thetas = np.zeros(self.K)
        for ii in range(self.K):
            thetas[ii] = np.random.beta(self.S[ii] + self.alpha, self.F[ii] + self.beta)

        # Draw arm with highest probability
        return np.argmax(thetas)

    def update(self, reward: int, optimal_action: int):
        if reward == 1:
            self.S[optimal_action] += 1
        else:
            self.F[optimal_action] += 1