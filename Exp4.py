from typing import Sequence

import numpy as np

from bandit import Bandit
from sklearn.multiclass import OneVsRestClassifier
class exp4(Bandit):
    def __init__(self,
                 T: int,
                 d: int,
                 M : int,
                 K: int,
                 eta: float,
                 gamma: float,
    ) -> None:
        super(exp4, self).__init__(T=T,d=d,K=K)
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.Q = np.repeat(1/M,M)

    def get_action(self,
                   context: np.array,
                   experts: Sequence[OneVsRestClassifier],
    ) -> int:
        E = np.zeros((self.M, self.K))
        for idx, e in enumerate(experts):
            E[idx,:] = e.predict_proba(context)
        P = self.Q.dot(E)
        action = np.random.choice(np.arange(50), size=1, p=P)

        return action, P, E

    def update(self,
               action: int,
               reward: int,
               E: np.array,
               P: np.array,
    ) -> None:
        r = np.ones(self.K)
        r[action] = 1 - (1 - reward)/(P[action] + self.gamma)
        r = E.dot(r)
        self.Q = np.exp(self.eta * r)*self.Q / (np.sum(np.exp(self.eta * r)*self.Q))

