import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import get_data, compute_regret


class LinUCB(object):

    def __init__(self, alpha, d, T, K, lambda_):
        self.alpha = alpha
        self.d = d
        self.T = T
        self.K = K
        self.lambda_ = lambda_
        self.A = lambda_ * np.identity(self.d)
        self.b = np.zeros((self.d))

    def get_action(self, mtx_content):
        """Compute S and Select action with max S """
        S = np.zeros(self.K)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv.dot(self.b)
        
        for ii in range(self.K):
            S[ii] = theta.dot(mtx_content[ii]) + self.alpha*np.sqrt(mtx_content[ii].T.dot(A_inv).dot(mtx_content[ii]))

        optimal_action = np.argmax(S)   
        return optimal_action

    def update(self, reward, mtx_content, optimal_action):
        """ Update matrix A and vector b."""
        self.A += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
        self.b += reward *  mtx_content[optimal_action]
        return self.A, self.b

if __name__ == '__main__':
    streaming_batch, user_feature, reward_list, action_context = get_data()
    action_context = np.array(action_context.iloc[:, 2:])
    K, D = action_context.shape
    print(K, D)
    T = len(streaming_batch)
    # T = 10000
    # print(len(streaming_batch))
    alpha = 0.3
    lambda_ = 1
    seq_error = np.full(T, 0)
    bandit = LinUCB(alpha, D, T, K, lambda_)

    for t in tqdm(range(0, T - 1)):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t + 1, 0])])
        context = feature_user * action_context
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t + 1, 0])]
        optimal_action = bandit.get_action(context)
        # optimal_action = random.randint(0, 49)

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

        bandit.update(reward, context, optimal_action)

    cumulative_regret = compute_regret(seq_error)
    x = [i for i in range(len(cumulative_regret))]
    plt.plot(x, cumulative_regret)
    plt.savefig("LinUCB.png")
