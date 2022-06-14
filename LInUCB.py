import numpy as np 

class LinUCB(object):

    def __init__(self, alpha, D, T, K, lambda_):
        self.alpha = alpha
        self.D = D
        self.T = T
        self.K = K
        self.lambda_ = lambda_
        self.A = lambda_ * np.identity(self.D)
        self.b = np.zeros((self.D))

    def get_action(self, mtx_content):
        """Compute S and Select action with max S """
        S = np.zeros(self.K)
        
        for ii in range(self.K):
            S[ii] = self.alpha*np.sqrt(mtx_content[ii].T.dot(np.linalg.inv(self.A)).dot(mtx_content[ii]))

        optimal_action = np.argmax(S)   
        return optimal_action

    def update(self, reward, mtx_content, optimal_action):
        """ Update matrix A and vector b."""
        self.A += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
        self.b += reward *  mtx_content[optimal_action]
        return self.A, self.b