import random

from matplotlib import pyplot as plt
from LInUCB import *
from Exp4 import exp4
from utils import get_data, training_data, train_expert, compute_regret


def main_exp4():
    streaming_batch, user_feature, reward_list, action_context = get_data()
    X, y = training_data(user_feature, reward_list, streaming_batch, 1000)
    experts = train_expert(X, y)
    T = 10000
    d = 18
    M = 2
    K = 50
    eta = 0.4
    gamma = 0.4
    bandit = exp4(T, d, M, K, eta, gamma)
    seq_error = np.full(T, 0)
    for t in range(T - 1):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t + 1, 0])])
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t + 1, 0])]
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
    return cumulative_regret, seq_error[T - 10:T + 10]

def main():

    streaming_batch, user_feature, reward_list, action_context = get_data()
    action_context = np.array(action_context.iloc[:,2:])
    K, D = action_context.shape
    print(K, D)
    T = len(streaming_batch)
    T = 10000
    #print(len(streaming_batch))
    alpha = 0.3
    lambda_ = 1
    seq_error = np.full(T,0)
    bandit = LinUCB(alpha, D, T, K, lambda_)

    for t in range(0, T-1):
        feature_user = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t+1, 0])])
        context = feature_user * action_context
        watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t+1, 0])]
        optimal_action = bandit._get_action(context)
        optimal_action = random.randint(0, 49)
        
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

        bandit._update(reward, context, optimal_action)
        
    
    cumulative_regret = compute_regret(seq_error)
    return cumulative_regret, seq_error[T-10:T+10]


if __name__ == "__main__":
    avg_r, r  = main_exp4()
    plt.plot(avg_r)
    plt.show()