import numpy as np
import scipy.linalg as lalg

class target(object):
    def __init__(self, param, dt):
        self.in_size = 7

        self.state = np.append(self.conv_param(param[:4]), [[0],[0],[0]], axis = 0) #[[x], [y], [h], [w/h], [x'], [y'], [h']] x.y -> center
        self.pred_state = self.state

        std = np.ones((self.in_size))
        self.cov = np.diag(np.square(std))
        self.pred_cov = self.cov

        self.dt = dt
        self.update_matrix = self.get_update_matrix(self.dt)

        self.observation_matrix = np.eye(self.in_size, self.in_size) #maybe just remove cause we dont need all the observation_matrix, or maybe adapt to fisheye?
        
    def get_update_matrix(self, dt):
        return np.array(
            [[1,0,0,0,dt,0,0],
            [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]], dtype=np.float32)
    
    def conv_param(self, param):
        return np.array([[param[0] + param[2]/2], [param[1] + param[3]/2], [param[3]], [param[2]/param[3]]])
    
    def center_to_corner(self, v):
        w = v[2] * v[3]
        h = v[3]
        return [v[0] - w/2, v[1] - h/2, w, h]

    def update_pred(self):
        self.pred_state = np.matmul(self.update_matrix, self.state) #No controled input
        self.pred_cov = np.matmul(np.matmul(self.update_matrix, self.cov), self.update_matrix.T) #+Q (ignored)
    
    def update_state(self, detect):
        R_cov_matrix = np.diag(np.ones((self.in_size)))
        R_cov_matrix *= (1 - detect[4])
        innovation_cov = np.matmul(np.matmul(self.observation_matrix, self.pred_cov), self.observation_matrix.T) + R_cov_matrix

        #Going through Choleski decomposition to invert matrix (need to understand this better) - Maybe dont need all the transpose if we take upper
        cho_factor, lower = lalg.cho_factor(innovation_cov, lower = True, check_finite = False)
        kalman_gain = lalg.cho_solve((cho_factor, lower), np.matmul(self.pred_cov, self.observation_matrix.T).T, check_finite=True).T

        param = self.conv_param(detect[:4])
        param = np.append(param, (param[:3] - self.state[:3])/self.dt, axis = 0)
        innovation_mean = param - np.matmul(self.observation_matrix, self.pred_state)

        #Update
        self.state = self.pred_state + np.matmul(kalman_gain, innovation_mean)
        print(self.state)
        self.cov = np.matmul(np.eye((self.in_size)) - np.matmul(kalman_gain, self.observation_matrix), self.pred_cov)