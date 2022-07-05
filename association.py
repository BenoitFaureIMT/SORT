# https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py
import numpy as np

track_indices = [] # 1 - ... - N
detect_indices = [] # 1 - ... - M
max_age = 10 # cascade Depth
t1 = 9.4877
t2 = 5.9915

mat_cov = np.identity(7)
pred_kal = [] #xywh liste des tracks prédits
bbox = [] #xywh liste des détections
min_cosine_dist = np.zeros((len(track_indices),len(detect_indices))) # Matrice N x M

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


# Step 1
cost_matrix = [[c(i,j) for i in range(len(track_indices))] for j in range(len(detect_indices))]
# Step 2
gate_matrix = [[b(i,j) for i in range(len(track_indices))] for j in range(len(detect_indices))]
# Step 3
matches = []
# Step 4
unmatched = detect_indices
# Step 5
for i in range(1,max_age):
    Tn = []
    for j in range(len(track_indices)):
        if j>0: # Changer condition : Tous les tracks non sélectionnés lors des n frames précédentes
            Tn.append(track_indices[j])
    X = min_cost_matching(cost_matrix,Tn,unmatched)
    for k in range(len(track_indices)):
        for l in range(len(detect_indices)):
            if gate_matrix[i][j]*X[i][j] > 0:
                matches.append((k,l))
    for m in range(len(detect_indices)): # Peut être optimisé en changeant l'odre des boucles
        if sum([gate_matrix[i][m]*X[i][m] for i in range(len(track_indices))]):
            unmatched.remove(m)

print(matches)
print(unmatched)