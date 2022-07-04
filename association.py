import numpy as np

track_indices = []
detect_indices = []
max_age = 1
t1 = 9.4877
t2 = 5.9915

def d1(i,j):
    pass

def d2(i,j):
    pass

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