import numpy as np 


class LinUCB(object):


    def __init__(self, alpha, d, K, T, lambda_, value):
        self.alpha = alpha
        self.d = d
        self.K = K
        self.T = T
        self.lambda_ = lambda_
        self.value = value

        

    def train(self, mtx_content):
        
        for t in range(self.T):
            A = self.lambda_*np.sparse.eye(self.d, dtype=np.int8)
            b = np.zeros((self.d, 1), dtype = np.int8)
            

        optimal_action = self._get_action(mtx_content,A,b)
        reward = self.get_reward(self.value, optimal_action, A, b)
        A, b = self._update(reward, mtx_content, optimal_action, A, b)
               
        




    def _get_action(self, mtx_content, A):
        """Compute S and Select action with max S """
        S = np.zeros(self.K, dtype = np.int8)
        
        for ii in range(self.K):
            val = np.dot(mtx_content[ii], mtx_content[ii])
            S[ii] = self.alpha*np.sqrt(mtx_content[ii].T.dot(np.linalg.inv(A)).dot(mtx_content[ii]))

        optimal_action = np.argmax(S)   
        
          
            
        return optimal_action


    def get_reward(self, value):
            reward = value
            return reward

    def _update(self, reward, mtx_content, optimal_action, A, b):
        """ Update matrix A and vector b."""
        A += mtx_content[optimal_action].dot(mtx_content[optimal_action].T)
        b += reward *  mtx_content[optimal_action]
        return A, b