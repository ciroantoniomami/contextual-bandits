import numpy as np 

class LinUCB(object):

    def __init__(self, alpha, D, T, K, lambda_, disjoint):
        self.alpha = alpha
        self.D = D
        self.T = T
        self.K = K
        self.lambda_ = lambda_
        self.disjoint = disjoint

        if not disjoint:
            self.A = lambda_ * np.identity(self.D)
            self.b = np.zeros((self.D))
        else:
            self.A = [lambda_ * np.identity(self.D) for _ in range(self.K)]
            self.b = [np.zeros((self.D)) for _ in range(self.K)]

    def get_action(self, mtx_content):
        """Compute S and Select action with max S """
        S = np.zeros(self.K)
        if self.disjoint:
            theta = [np.linalg.inv(self.A[ii]).dot(self.b[ii]) for ii in range(self.K)]
        else:
            theta = np.linalg.inv(self.A).dot(self.b)
        
        if self.disjoint:
            for ii in range(self.K):
                S[ii] += mtx_content[ii].dot(theta[ii])
                S[ii] += self.alpha*np.sqrt(mtx_content[ii].T.dot(np.linalg.inv(self.A[ii])).dot(mtx_content[ii]))
        else:
            for ii in range(self.K):
                S[ii] += mtx_content[ii].dot(theta)
                S[ii] += self.alpha*np.sqrt(mtx_content[ii].T.dot(np.linalg.inv(self.A)).dot(mtx_content[ii]))

        optimal_action = np.argmax(S)   
        return optimal_action

    def update(self, reward, mtx_content, optimal_action):
        """ Update matrix A and vector b."""
        if self.disjoint:
            self.A[optimal_action] += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
            self.b[optimal_action] += reward *  mtx_content[optimal_action]
        else:
            self.A += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
            self.b += reward *  mtx_content[optimal_action]
        return self.A, self.b