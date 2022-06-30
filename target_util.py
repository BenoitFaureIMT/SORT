import numpy as np

class target(object):
    def __init__(self, param):
        self.state = np.append(param.T, [[0],[0],[0]])
        self.pred_state = self.state
        self.cov = np.array([[0],[0],[0],[0],[1],[1],[1]])
        self.pred_cov = self.cov

        dt = 0.1
        self.update_matrix = np.append(
            [[1,0,0,0,dt,0,0],
            [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]], dtype=np.float32)
    
    def update_pred(self):
        self.pred_state = np.matmul(self.update_matrix, self.state) #No controled input
        self.pred_cov = np.matmul(np.matmul(self.update_matrix, self.cov), self.update_matrix.T) #+Q (ignored)
    
    def update_state(self, detect):
