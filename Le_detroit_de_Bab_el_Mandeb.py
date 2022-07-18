#Dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import cv2

from Les_Communistes import Aquaman, Atlante

#Class to train the CNN of the Aquaman filter
class Aquagym(object):
    def __init__(self, keras_model):
        self.model = keras_model

    #Function to run the training
    #       - inputs : detections, realPositions, dts (detections need to correspond (same index) through out the frames)
    #       detections -> [ [[bbox], bboxes], frames ]
    #       realPositions -> same
    #       dts -> [dt_frame1-2, dt_frame2-3 ...]
    def run(self, width, height, detections, realPositions, dts, missed_detections = 0, display = False):
        if(min(len(detections), len(realPositions), len(dts)) < 2):
            print("I can't do shit to train if there is less than 2 frames you fucking retard!!!!!")
            print("Go fuck yourself")
            return
        
        AFilter = Aquaman(width, height, "")
        AFilter.coeff_model = self.model
        AFilter.pre_calc(1) #TODO choose increment
        targs = [Atlante(d) for d in detections[0]]

        if missed_detections == 0:
            missed_detections = np.zeros((len(detections), len(detections[0])))
    
        for i in range(1, len(detections)):

            inps = []
            ideals = []

            #Displaying results ish
            if display:
                img = np.zeros((width, height, 3))

            for j in range(len(targs)):

                AFilter.pred_next_state(targs[j], dts[i])
                if missed_detections[i][j] == 0:
                    err, inp = AFilter.update_state(targs[j], detections[i][j], dts[i])
                    inps.append(inp[0])
                    ideal = np.divide(np.array([realPositions[i][j]]).T - targs[j].pred_state, err, out = np.zeros_like(err), where=err!=0)
                    ideals.append(ideal.T[0])
                else:
                    AFilter.update_state_no_detection(targs[j], dts[i])

                #Displaying
                if display:
                    self.draw_rect(img, realPositions[i][j], (255, 255, 255))
                    self.draw_rect(img, targs[j].state.T[0], (0, 255, 0) if missed_detections[i][j] == 0 else (0, 0, 255))

            if len(inps) > 0:
                self.model.fit(np.array(inps), np.array(ideals), epochs=1, batch_size=1)

            if display:
                cv2.putText(img, str(i), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Le detroit de Bab-el-Mandeb", img)
                k = cv2.waitKey(1)
                if k == 27:
                    exit()
    
    #Function to save model
    def save(self, save_path):
        weight = []
        bias = []
        for l in self.model.layers:
            a = l.get_weights()
            weight.append(a[0])
            bias.append(a[1])
        np.save(save_path, [weight, bias], allow_pickle = True)

    #Utils for drawing
    def draw_rect(self, img, state, color):
        cv2.rectangle(img, (int(state[0] - state[2]/2), int(state[1] - state[3]/2)), (int(state[0] + state[2]/2), int(state[1] + state[3]/2)), color, 2)

theta = 0
dt = 0.1
w, h = 600, 600
r = 200
rs_list = []
ds_list = []
dt_list = []
missed = []
for f in range(500):
    d_theta = (2*np.pi/20 * dt) % (2 * np.pi)
    theta += d_theta
    rs_list.append([np.array([r * np.sin(theta) + w/2, r * np.cos(theta) + h/2, 20 * (1 + np.sin(theta) / 10), 40 * (1 + np.sin(theta) / 10), 
    r * d_theta / dt * np.cos(theta), -r * d_theta / dt * np.sin(theta), 20 * np.cos(theta) / 10 * d_theta / dt, 40 * np.cos(theta) / 10 * d_theta / dt])])
    ds_list.append([rs_list[-1][0][:4] + (np.random.rand(4) - 0.5) * r * 0.05])
    dt_list.append(dt)
    missed.append([1 if f > 100 and np.random.rand() < 0.5 else 0])

k_model = Sequential()
k_model.add(Dense(4, input_shape = (3,), activation='sigmoid'))
k_model.add(Dense(8, input_shape = (4,), activation='sigmoid'))
k_model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())

gym = Aquagym(k_model)
gym.run(w, h, ds_list, rs_list, dt_list, missed_detections = missed, display = True)
print("Run Done")
gym.save("aqua_weights_test")