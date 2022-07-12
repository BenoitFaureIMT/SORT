from email.mime import base
from random import random
from tkinter import N
from xml.sax.handler import DTDHandler
from cv2 import cornerEigenValsAndVecs
import numpy as np
import cv2
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

screen_w = 1000
screen_h = 1000
center = np.array([[screen_w/2, screen_h/2]]).T
base_r = 20
real_state = np.array([[screen_w/2, screen_h/2, base_r, 0, 0, 0]]).T
t = 0
dt = 0.1

acc_dir = 1
acc = 500

class cust_model(object):
    def __init__(self, keras_weights):
        print(keras_weights)
        self.weights = []
        self.bias = []
        for l in keras_weights:
            self.weights.append(l[0])
            self.bias.append(l[1])
        self.n = len(self.weights)
    
    def sigmoid(self, inp):
        exp = np.exp(inp)
        return exp / (1 + exp)
    
    def predict(self, inp):
        for i in range(self.n):
            inp = np.matmul(inp, self.weights[i]) + self.bias[i]
            inp = self.sigmoid(inp)
        return inp

class Kalman(object):
    def __init__(self, base_state, dt):
        base_state = base_state.T[0]
        self.state = np.array([[base_state[0], base_state[1], base_state[2], 0, 0, 0]]).T
        self.pred_state = self.state
        self.motion_matrix = self.get_motion_matrix(dt)
        # self.coeff_matrix = np.random.rand(1,7)
        self.coeff_model = Sequential()
        self.coeff_model.add(Dense(3, input_shape = (3,), activation='sigmoid'))
        self.coeff_model.add(Dense(6, input_shape = (3,), activation='sigmoid'))
        self.coeff_model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())

        self.missed_detection = False
        self.last_dected_state = self.state
        self.elapsed_time_detection = 0

    def get_motion_matrix(self, dt):
        return np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
    
    def get_weights(self):
        self.coeff_model = cust_model([l.get_weights() for l in self.coeff_model.layers])
 
    def pred_next_state(self):
        self.pred_state = np.matmul(self.motion_matrix, self.state)
    
    def update_state_no_detect(self, dt):
        self.motion_matrix = self.get_motion_matrix(dt)
        self.state = self.pred_state

        self.missed_detection = True
        self.elapsed_time_detection += dt
    
    def update_state_no_train(self, detect, dt):
        #t_a = time.perf_counter()

        self.motion_matrix = self.get_motion_matrix(dt)
        detect = detect.T[0]
        n_det = np.array([[detect[0], detect[1], detect[2], (detect[0] - self.state[0,0]) / dt, (detect[1] - self.state[1,0]) / dt, (detect[2] - self.state[2,0]) / dt]]).T
        
        #t_b = time.perf_counter()
        
        if(self.missed_detection):
            self.elapsed_time_detection += dt
            n_det = np.array([[detect[0], detect[1], detect[2], 
            (detect[0] - self.last_dected_state[0]) / self.elapsed_time_detection, 
            (detect[1] - self.last_dected_state[1]) / self.elapsed_time_detection, 
            (detect[2] - self.last_dected_state[2]) / self.elapsed_time_detection]]).T
        err = n_det - self.pred_state

        #t_c = time.perf_counter()

        inp = np.array([[(self.pred_state[0,0] - detect[0]) / screen_w, (self.pred_state[1,0] - detect[1]) / screen_h, 0 if self.missed_detection else 1]]).T
        #print(self.coeff_model.predict(inp.T))

        #t_d = time.perf_counter()

        self.state = self.pred_state + self.coeff_model.predict(inp.T).T * err
        print(self.coeff_model.predict(inp.T))

        #t_e = time.perf_counter()

        self.missed_detection = False
        self.last_dected_state = detect
        self.elapsed_time_detection = 0

        #t_f = time.perf_counter()

        #print(t_b - t_a, t_c - t_b, t_d - t_c, t_e - t_d, t_f - t_e)


    
    def update_state(self, detect, dt, r_state):
        #t_a = time.perf_counter()
        self.motion_matrix = self.get_motion_matrix(dt)
        detect = detect.T[0]
        n_det = np.array([[detect[0], detect[1], detect[2], (detect[0] - self.state[0,0]) / dt, (detect[1] - self.state[1,0]) / dt, (detect[2] - self.state[2,0]) / dt]]).T
        #t_b = time.perf_counter()
        if(self.missed_detection):
            self.elapsed_time_detection += dt
            n_det = np.array([[detect[0], detect[1], detect[2], 
            (detect[0] - self.last_dected_state[0]) / self.elapsed_time_detection, 
            (detect[1] - self.last_dected_state[1]) / self.elapsed_time_detection, 
            (detect[2] - self.last_dected_state[2]) / self.elapsed_time_detection]]).T
        err = n_det - self.pred_state
        #t_c = time.perf_counter()
        #inp = np.array([[self.state[0,0]/320, self.state[1,0]/320, self.state[2,0]/20, detect[0]/320, detect[1]/320, detect[2]/20, 0.8]]).T
        #inp = np.random.rand(1,1) * .2 + .7
        inp = np.array([[(self.pred_state[0,0] - detect[0]) / screen_w, (self.pred_state[1,0] - detect[1]) / screen_h, 0 if self.missed_detection else 1]]).T
        #print(self.coeff_model.predict(inp.T))
        # coeff_1 = np.matmul(self.coeff_matrix, inp)[0,0]
        # coeff = 1 / (1 + coeff_1)
        #t_d = time.perf_counter()
        # self.state = self.state + coeff * err
        self.state = self.pred_state + self.coeff_model.predict(inp.T).T * err
        ideal_coeff = (r_state - self.pred_state)/err
        self.coeff_model.fit(inp.T, ideal_coeff.T, epochs=1, batch_size=1)

        #t_e = time.perf_counter()

        self.missed_detection = False
        self.last_dected_state = detect
        self.elapsed_time_detection = 0

        #t_f = time.perf_counter()

        #print(t_b - t_a, t_c - t_b, t_d - t_c, t_e - t_d, t_f - t_e)

        # real_c = np.average(((r_state - self.state)/err).T[0])
        # grad = inp * 1 / ((1 + coeff_1)**2) * (coeff - real_c) * 2
        # print(real_c, coeff)
        # self.coeff_matrix -= grad.T



targ = Kalman(real_state, t)
count = 0

while True:
    t = time.perf_counter()
    img = np.zeros((screen_h, screen_w, 3), np.uint8)
    print("--------------------------")

    #Updating position
    real_state[:2] += real_state[3:5] * dt
    #Updating speed
    v = max([np.abs(real_state[0][0] - screen_w / 2) * 2 / screen_w, np.abs(real_state[1][0] - screen_h / 2) * 2 / screen_h])
    acc_dir = (acc_dir + 2 * np.pi * (np.random.rand() + 0.5) / 10 * dt) % (2 * np.pi)
    a = np.array([[np.cos(acc_dir), np.sin(acc_dir)]]).T * acc * dt * (1 + np.random.rand())

    corr = center - real_state[:2]
    b = -corr 

    real_state[3:5] += a * (1 - v) ** 1/5 - b * v ** 5
    #Updating rad
    real_state[2] = base_r * (1 - v/5) ** 2
    real_state[5] = (1 - v/5) ** 2

    #Creating detection
    detected_state = real_state * (1 + (np.random.rand(6, 1) - 0.5) * np.array([[0.05, 0.05, 0.1, 0, 0, 0]]).T * base_r/20)
    #Updating kalman
    noDetection = count > 60 and np.random.rand(1)[0] < 0.3 # and count % 10 % 2 == 0
    if(count == 201):
        targ.get_weights()
    noTraining = count > 200
    t_track = time.perf_counter()
    if(noDetection):
        print("######NO DETECTION######")
        targ.update_state_no_detect(dt)
    else:
        if (noTraining):
            targ.update_state_no_train(detected_state, dt)
        else:
            targ.update_state(detected_state, dt, real_state)
    targ.pred_next_state()
    t_track = time.perf_counter() - t_track
    #print(targ.state)

    #Show real object
    cv2.circle(img, (int(real_state[0][0]), int(real_state[1][0])), int(real_state[2][0]), (255, 255, 255), 2)
    #Show detection
    # cv2.circle(img, (int(detected_state[0][0]), int(detected_state[1][0])), int(detected_state[2][0]), (0, 255, 0), 1)
    cv2.circle(img, (int(targ.last_dected_state[0]), int(targ.last_dected_state[1])), int(targ.last_dected_state[2]), (0, 255, 0), 1)
    #Show position
    cv2.circle(img, (int(targ.state[0][0]), int(targ.state[1][0])), int(targ.state[2][0]), (255, 255, 0), 2)
    #Show predition
    cv2.circle(img, (int(targ.pred_state[0][0]), int(targ.pred_state[1][0])), int(targ.pred_state[2][0]), (0, 0, 255), 1)

    #Show info
    cv2.putText(img, str(count) + (" STOPED TRAINING" if noTraining else ""), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(int(1/dt)) + " | " + str(int(1/t_track)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(noDetection):
        cv2.putText(img, str("NO DETECTION"), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    #Show random acceleration vector
    # d_disp = real_state[:2] + a
    # cv2.line(img, (int(real_state[0][0]), int(real_state[1][0])), (int(d_disp[0][0]), int(d_disp[1][0])), (255, 0, 0))
    
    cv2.imshow("La libye à la coupe du monde 2014", img)
    k = cv2.waitKey(1)
    dt = time.perf_counter() - t
    if k == 27:
        break
    count += 1

cv2.destroyAllWindows()
