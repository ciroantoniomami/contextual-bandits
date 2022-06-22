import random
import numpy as np
import time
import pickle

import matplotlib.pyplot as plt

from random import randint
from sklearn.linear_model import LogisticRegression
from utils import get_data, compute_regret


class EpsilonGreedy(object):

    def __init__(self, K: int, eps: float, decay: float, fit_step: int) -> None:

        self.K = K
        self.eps = eps
        self.decay = decay
        self.oracles = {k: LogisticRegression() for k in range(self.K)}
        self.data = {}
        self.fit_step = fit_step
        self.fitted = False

    def explore(self) -> int:
        return randint(0, self.K - 1)

    def update(self, context: np.array, action: int, reward: int, t: int) -> None:
        self.eps = self.eps * self.decay

        if action not in self.data.keys():
            self.data[action] = (context, np.ones(1) * reward)

        else:
            x, r = self.data[action]

            self.data[action] = (np.concatenate((x, context)), np.concatenate((r, np.ones(1) * reward)))

        if len(self.data.keys()) == self.K and np.min([len(set(v[1])) for v in self.data.values()]) > 1:

            if t % self.fit_step == 0 or self.fitted == False:
                for k in self.oracles.keys():
                    self.oracles[k].fit(self.data[k][0], self.data[k][1])
                self.fitted = True
        return

    def exploit(self, context: np.array) -> int:
        return np.argmax([oracle.predict_proba(context)[0][1] for oracle in self.oracles.values()])

    def get_action(self, context: np.array, t: int) -> int:
        if random.uniform(0, 1) > self.eps and len(self.data.keys()) == self.K and np.min(
                [len(set(v[1])) for v in self.data.values()]) > 1:
            return self.exploit(context)

        else:
            return self.explore()


if __name__ == "__main__":
    streaming_batch, user_feature, reward_list, action_context = get_data()
    action_context = np.array(action_context.iloc[:, 2:])
    K, _ = action_context.shape
    start_T = 5000
    T = len(streaming_batch)
    tic = time.perf_counter()
    decay = 0.9999
    eps = 0.1
    fit_step = 100

    bandit = EpsilonGreedy(K, eps, decay, fit_step)

    seq_error = np.full(T - start_T, 0)
    seq_reward = np.full(T - start_T, 0)

    for t in range(0, T - 1 - start_T):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t + start_T + 1, 0])])
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t + start_T + 1, 0])]
        optimal_action = bandit.get_action(feature_user, t)

        if optimal_action not in list(watched_list['movie_id']):
            reward, regret = 0.0, 1.0

        else:
            reward, regret = 1.0, 0.0

        bandit.update(feature_user, optimal_action, reward, t)

        if t == 0:
            seq_error[t] = regret
            seq_reward[t] = reward
        else:
            seq_error[t] = seq_error[t - 1] + regret
            seq_reward[t] = reward

    toc = time.perf_counter()
    print(f"Running EpsilonGreedy in {toc - tic:0.4f} seconds")

    cumulative_regret = compute_regret(seq_error)
    print(len(cumulative_regret))

    with open("regret_egreedy_{e}_fit{f}.pkl".format(e=eps, f=fit_step), 'wb') as f:
        pickle.dump(cumulative_regret, f)

    x = [i for i in range(len(cumulative_regret[:-1]))]
    plt.plot(x, cumulative_regret[:-1])
    plt.savefig("Egreedy_{e}_fit{f}.png".format(e=eps, f=fit_step))
