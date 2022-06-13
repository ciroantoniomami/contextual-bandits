import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


def get_data():
    streaming_batch = pd.read_csv('data/streaming_batch.csv', sep='\t', header=0,names=['user_id'], engine='c')
    user_feature = pd.read_csv('data/user_feature.csv', sep='\t', header=0,index_col=0, engine='c')
    actions_id = list(pd.read_csv('data/actions.csv', sep='\t', header=0,engine='c')['movie_id'])
    reward_list = pd.read_csv('data/reward_list.csv', sep='\t', header=0,engine='c')
    action_context = pd.read_csv('data/action_context.csv', sep='\t', engine='c')
    return streaming_batch, user_feature, reward_list, action_context

def compute_regret(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret

def training_data(
                  user_feature: pd.DataFrame,
                  reward_list: pd.DataFrame,
                  streaming_batch: pd.DataFrame,
                  num_samples: int,
)-> (np.array, np.array):
    user = np.unique(np.array(streaming_batch))
    samples = np.random.choice(user, size=num_samples, replace=True)
    X = []
    y = []
    for idx in samples:
        feature_user = np.array(user_feature[user_feature.index == idx]).squeeze(0)
        watched_list = reward_list[reward_list['user_id'] == idx]
        if len(watched_list) == 0:
            continue
        X.append(feature_user)
        y.append(int(watched_list['movie_id'].sample()))

    return  np.array(X), np.array(y)

def train_expert(X, y):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB())
    logreg.fit(X, y)
    mnb.fit(X, y)
    return [logreg, mnb]