import numpy as np
from scipy.optimize import linear_sum_assignment

from ReID import extract_cost_matrix

class association_algo(object):
    def __init__(self):
        self.track_indices = []
        self.detect_indices = []
        self.mat_cov = np.identity(7) # A retaper
        self.predictions = [] #xywh liste des tracks prédits
        self.tracks = []
        self.detections = [] #xywh liste des détections
        # Trucs de ReId
        self.target = []
        self.detection_feature = []
        # Trucs de ReId
        self.t1 = 9.4877
        self.t2 = 5.9915
        self.lmbd = 0.5
        self.max_age = 10

    def d1(self, i, j):
        mat = self.detections[:][j]-self.predictions[i]
        return np.dot(np.dot(np.transpose(mat),self.mat_cov),mat)
    
    def d2(self, i, j):
        return extract_cost_matrix(self.target, self.detection_feature)
    
    def b(self, i, j):
        return np.dot(self.b1(i,j),self.b2(i,j))
    
    def b1(self, i, j):
        if self.d1(self, i, j, self.detections, self.predictions, self.mat_cov) <= self.t1:
            return 1
        else:
            return 0
    
    def b2(self, i, j):
        if self.d2(self, i, j, self.target, self.detection_feature) <= self.t2:
            return 1
        else:
            return 0
    
    def c(self, i, j):
        return self.lmbd*self.d1(self, i, j, self.detections, self.predictions, self.mat_cov)+(1-self.lmbd)*self.d2(self, i, j, self.target, self.detection_feature)
    
    def dist_metric(self, func_c, track_ind, detect_ind):
        return [[func_c(i,j) for i in range(len(track_ind))] for j in range(len(detect_ind))]

    def hungarian(self, func_dist_metric, T, U, max_dist=0.9):
        if len(U) == 0 or len(T) == 0:
            return [], T, U
        cost_matrix = func_dist_metric(self, self.c, T, U)
        cost_matrix[cost_matrix > max_dist] = max_dist + 1e-5 # Coeff a potentiellement changer
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(U):
            if col not in col_indices:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(self.track_indices):
            if row not in row_indices:
                unmatched_tracks.append(track_idx)
        for row, col in zip(row_indices, col_indices):
            track_idx = self.track_indices[row]
            detection_idx = U[col]
            if cost_matrix[row, col] > max_dist:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections
    
    def match_cascade(self):
        N = len(self.track_indices)
        M = len(self.detect_indices)
        gate_matrix = [[self.b(i,j) for i in range(N)] for j in range(M)]
        matches = []
        unmatched = self.detect_indices
        for i in range(1,self.max_age):
            Tn = []
            for j in range(N):
                if j>0: # Changer condition : Tous les tracks non sélectionnés lors des n frames précédentes
                    Tn.append(self.track_indices[j])
            X = self.hungarian(self, self.dist_metric, self.tracks, self.detections, Tn, unmatched, max_dist=0.9)[0]
            for k in range(N):
                for l in range(M):
                    if gate_matrix[i][j]*X[i][j] > 0:
                        matches.append((k,l))
            for m in range(M): # Peut être optimisé en changeant l'odre des boucles
                if sum([gate_matrix[i][m]*X[i][m] for i in range(N)]):
                    unmatched.remove(m)
        return matches, unmatched