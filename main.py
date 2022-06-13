import pandas as pd
import numpy as np
import random
from LInUCB import *


def get_data():
    streaming_batch = pd.read_csv('data/streaming_batch.csv', sep='\t', names=['user_id'], engine='c')
    user_feature = pd.read_csv('data/user_feature.csv', sep='\t', header=0,index_col=0, engine='c')
    actions_id = list(pd.read_csv('data/actions.csv', sep='\t', header=0,engine='c')['movie_id'])
    reward_list = pd.read_csv('data/reward_list.csv', sep='\t', header=0,engine='c')
    action_context = pd.read_csv('data/action_context.csv', sep='\t', engine='c')
    return streaming_batch, user_feature, reward_list, action_context

def compute_regret(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret

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
        optimal_action = bandit.get_action(context)
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

        bandit.update(reward, context, optimal_action)
    
    cumulative_regret = compute_regret(seq_error)
    return cumulative_regret, seq_error[T-10:T+10]


if __name__ == "__main__":
    avg_r, r  = main()
    print(r)