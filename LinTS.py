
from __future__ import annotations
import numpy as np
from main import get_data, compute_regret

class LinTS(object):

    def __init__(self, R: float, epsilon: float, delta: float, D: int, K: int) -> None:
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"Epsilon should be in (0,1). Passed parameter was {epsilon}")
        if delta <= 0 or delta > 1:
            raise ValueError(f"Delta should be in (0,1]. Passed parameter was {delta}")

        self.R = R
        self.epsilon = epsilon
        self.delta = delta
        self.D = D
        self.B = np.identity(D)
        self.mu = np.zeros(D)
        self.f = np.zeros(D)
        self.K = K

    def get_action(self, mtx_content: np.array,  t_step: int):
        v = self.R * np.sqrt(24 / self.epsilon * self.D * np.log((t_step+1) / self.delta))
        cov_matrix = v**2 * np.linalg.inv(self.B)

        u = np.random.multivariate_normal(mean=self.mu, cov=cov_matrix, size=1).reshape(self.D, 1)

        print(cov_matrix)

        S = np.zeros(self.K)
        for ii in range(self.K):
            S[ii] = mtx_content[ii].dot(u)
        
        return np.argmax(S)

    def update(self, reward: int, mtx_content: np.array, optimal_action: int):
        self.B += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
        self.f += reward * mtx_content[optimal_action]
        self.mu = np.linalg.inv(self.B).dot(self.f)
    
if __name__ == "__main__":

    streaming_batch, user_feature, reward_list, action_context = get_data()
    action_context = np.array(action_context.iloc[:,2:])
    K, D = action_context.shape
    T = len(streaming_batch)
    T = 10000

    delta = 0.1
    R = 0.3
    epsilon = 1 / np.log(T)

    bandit = LinTS(R=R, epsilon=epsilon, delta=delta, D=D, K=K)

    seq_error = np.full(T,0)
    for t in range(0, T-1):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t+1, 0])])
        context = feature_user * action_context
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t+1, 0])]
        optimal_action = bandit.get_action(mtx_content=context, t_step=1)
        
        if optimal_action not in list(watched_list['movie_id']):
            reward = 0.0
            regret = 1.0
        else:
            reward = 1.0
            regret = 0.0

        if t == 0:
                seq_error[t] = regret
        else:
                seq_error[t] = seq_error[t-1] + regret

        bandit.update(reward, context, optimal_action)
    
    cumulative_regret = compute_regret(seq_error)
    print(seq_error[-10:])