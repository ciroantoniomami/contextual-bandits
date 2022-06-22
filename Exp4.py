from typing import Sequence

import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import training_data, train_expert, compute_regret, get_data


class exp4(object):
    def __init__(self,
                 T: int,
                 d: int,
                 M : int,
                 K: int,
                 eta: float,
                 gamma: float,
    ) -> None:
        self.T = T
        self.d = d
        self.K = K
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
            if np.sum(e.predict(context)) > 0:
                E[idx,:] = e.predict(context)/np.sum(e.predict(context))
            else:
                E[idx, :] = np.repeat(1/self.K,self.K)
        P = self.Q.dot(E)
        action = np.random.choice(np.arange(self.K), size=1, p=P)

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

    def expert_regret(self,
                      r: np.array,
                      E: np.array,
    )-> float:
        return np.max(E.dot(r))


if __name__ == '__main__':
    streaming_batch, user_feature, reward_list, action_context = get_data()
    X, y = training_data(user_feature, reward_list, streaming_batch, 5000)
    experts = train_expert(X, y)
    streaming_batch = streaming_batch.iloc[5000:]
    # T = len(streaming_batch)
    T = 5000
    d = 18
    M = 3
    K = 10
    eta = 0.001
    gamma = 0
    bandit = exp4(T, d, M, K, eta, gamma)
    seq_error = np.full(T, 0)
    for t in tqdm(range(T - 1)):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t + 1, 0])])
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t + 1, 0])]
        r = np.zeros(K)
        r[watched_list['movie_id']] = 1
        optimal_action, P, E = bandit.get_action(feature_user, experts)
        if optimal_action not in list(watched_list['movie_id']):
            reward = 0.0
            regret = 1.0
        else:
            reward = 1.0
            regret = 0.0

        if t == 0:
            seq_error[t] = regret
        else:
            seq_error[t] = seq_error[t - 1] + regret

        bandit.update(optimal_action, reward, E, P)

    cumulative_regret = compute_regret(seq_error)
    x = [i for i in range(len(cumulative_regret))]
    plt.plot(x, cumulative_regret)
    plt.savefig("Exp4.png")
