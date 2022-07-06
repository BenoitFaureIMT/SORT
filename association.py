# https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py
import numpy as np
from scipy.optimize import linear_sum_assignment

track_indices = [] # 1 - ... - N
detect_indices = [] # 1 - ... - M
max_age = 10 # cascade Depth
t1 = 9.4877
t2 = 5.9915

mat_cov = np.identity(7)
pred_kal = [] #xywh liste des tracks prédits
bbox = [] #xywh liste des détections
min_cosine_dist = np.zeros((len(track_indices),len(detect_indices))) # Matrice N x M
tracks = []
detections = [] #xywh liste des détections

def d1(i,j):
    mat = bbox[:][j]-pred_kal[i]
    return np.dot(np.dot(np.transpose(mat),mat_cov),mat)

def d2(i,j):
    return min_cosine_dist

def b(i,j):
    return np.dot(b1(i,j),b2(i,j))

def b1(i,j):
    if d1(i,j) <= t1:
        return 1
    else:
        return 0

def b2(i,j):
    if d2(i,j) <= t2:
        return 1
    else:
        return 0

def c(i,j,lmbd):
    return lmbd*d1(i,j)+(1-lmbd)*d2(i,j)





def min_cost_matching(func_dist_metric, track, detection, T, U, max_dist=0.9):
    if len(U) == 0 or len(T) == 0:
        return [], T, U
    cost_matrix = func_dist_metric(c, T, U)
    cost_matrix[cost_matrix > max_dist] = max_dist + 1e-5 # Coeff a potentiellement changer
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(U):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = U[col]
        if cost_matrix[row, col] > max_dist:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections







def dist_metric(func_c, track_ind, detect_ind):
    return [[func_c(i,j) for i in range(len(track_ind))] for j in range(len(detect_ind))]

# Matching Cascade
#cost_matrix = [[c(i,j) for i in range(len(track_indices))] for j in range(len(detect_indices))]
gate_matrix = [[b(i,j) for i in range(len(track_indices))] for j in range(len(detect_indices))]
matches = []
unmatched = detect_indices
for i in range(1,max_age):
    Tn = []
    for j in range(len(track_indices)):
        if j>0: # Changer condition : Tous les tracks non sélectionnés lors des n frames précédentes
            Tn.append(track_indices[j])
    X = min_cost_matching(dist_metric, tracks, detections, Tn,unmatched, 0.9)[0]
    for k in range(len(track_indices)):
        for l in range(len(detect_indices)):
            if gate_matrix[i][j]*X[i][j] > 0:
                matches.append((k,l))
    for m in range(len(detect_indices)): # Peut être optimisé en changeant l'odre des boucles
        if sum([gate_matrix[i][m]*X[i][m] for i in range(len(track_indices))]):
            unmatched.remove(m)

print(matches)
print(unmatched)