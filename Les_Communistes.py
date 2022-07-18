#Dependences
import numpy as np
import cv2
import time

#Custom adaptation of the Kalman filter - which is not a Kalman filter at all
#       -> Replaces Kalman's inovation with a NN
#Requirements for targets
#   -> state = [yc, xc, h, w, x', y', ?]
class Aquaman(object):
    def __init__(self, screen_w, screen_h, weight_path):
        #Gradient params
        self.d = 1

        #CNN params
        self.coeff_model = self.FCNN(weight_path)

        #Motion params
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.S = screen_w / 2
        self.grad = []
        self.incr = -1

    #--------------------------Motion fucntions--------------------------
    #Calculate gradient at pixel point px,py : returns [dpx/dx, dpx/dy, dpy/dx, dpy/dy] / S where S in the radius of the fisheye image in pixels
    def calc_gradient(self, px, py): #TODO Pray that this is correct
        return [1,0,1,0]
        px = px - self.screen_w/2
        py = py - self.screen_h/2
        normpxpy = np.sqrt(px*px + py*py)
        
        if (normpxpy < 1e-10):
            return [1, 1, 1, 1] #TODO Maybe choose better values
        elif (normpxpy > self.screen_w/2):
            return [0, 0, 0, 0]

        normxy = self.d*np.tan(np.arcsin(2*normpxpy/self.screen_w))
        x = px*normxy/normpxpy
        y = np.sqrt(max(0, normxy**2 - x**2)) * np.sign(py)

        n2 = normxy**2
        n3 = normxy**3

        A1 = np.arctan(normxy/self.d)
        A = np.sin(A1)
        A2 = np.cos(A1)/(1 + n2)
        dAdx = A2*x/(normxy*self.d)
        dAdy = A2*y/(normxy*self.d)

        B = x/normxy
        dBdx = 1/normxy - B*x
        dBdy = -1*x*y/(n3)

        C = np.sin(np.arccos(max(-1, min(1, x/normxy))))
        C1 = 0
        if (y > 1e-10):
            C1 = x/(np.sqrt(n2 - x**2))
        dCdx = (x**2/n3 - 1/normxy)*C1
        dCdy = x*y/n3*C1

        return [dAdx*B + A*dBdx, dAdy*B + A*dBdy, dAdx*C + A*dCdx, dAdy*C + A*dCdy]
    
    #Calculate matrix of gradient - option save -> save it at path save_path
    def pre_calc(self, incr, save_path = "", d = -1):
        if d > 0:
            self.d = d
        
        self.incr = incr

        dim = (int(self.screen_w/incr), int(self.screen_h/incr), 4)
        arr = np.zeros(dim)
        for x in range(dim[0]):
            for y in range(dim[1]):
                arr[x, y] = np.array(self.calc_gradient(x*incr, y*incr))
        
        #Save?
        if save_path != "":
            np.save(save_path, arr, allow_pickle = True)
        
        self.grad = arr #* self.S #TODO why the fuck does it work better without self.S
    
    #Load pre_calc matrix
    def load_pre_calc(self, save_path):
        self.grad = np.load(save_path, allow_pickle = True) #* self.S
        self.incr = self.screen_w / self.grad.shape[0]
    
    #Function to get gradient
    #   - Values are clamped btw 0 and max
    def get_gradient(self, state):
        pc = (np.clip(int(state[1] / self.incr), 0, self.screen_w - 1), 
        np.clip(int(state[0] / self.incr), 0, self.screen_h - 1))
        p1 = (np.clip(int((state[1] - state[3]/2) / self.incr), 0, self.screen_w - 1), 
        np.clip(int((state[0] - state[2]/2) / self.incr), 0, self.screen_h - 1))
        p2 = (np.clip(int((state[1] + state[3]/2) / self.incr), 0, self.screen_w - 1), 
        np.clip(int((state[0] + state[2]/2) / self.incr), 0, self.screen_h - 1))
        return [self.grad[pc[0], pc[1]], self.grad[p2[0], p2[1]] - self.grad[p1[0], p1[1]]]


    #Get motion matrix with gradients already in place TODO Does this shit work?
    def get_motion_matrix(self, state, targ, dt):
        grad = self.get_gradient(state)
        targ.last_grad = grad
        return np.array([
            [1, 0, 0, 0, dt * grad[0][0], dt * grad[0][1], 0, 0],
            [0, 1, 0, 0, dt * grad[0][2], dt * grad[0][3], 0, 0],
            [0, 0, 1, 0, dt * grad[1][0], dt * grad[1][1], dt, 0],
            [0, 0, 0, 1, dt * grad[1][2], dt * grad[1][3], 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]])

    #--------------------------NN functions--------------------------
    #   NN class
    #       -> The network is fully connected and small => nothing complicated
    #       -> Keras has to many things slowing down calculations for such a small network
    #       -> So we make a custom feedforward class
    #       -> Obligatory sigmoid because I am lazy
    class FCNN(object):
        def __init__(self, weight_path):
            if(weight_path == ""):
                return None
            all = np.load(weight_path, allow_pickle = True)
            self.weights = all[0]
            self.bias = all[1]
            self.n = len(self.weights)
        
        def sigmoid(self, inp):
            exp = np.exp(inp)
            return exp / (1 + exp)
        
        def predict(self, inp):
            for i in range(self.n):
                inp = np.matmul(inp, self.weights[i]) + self.bias[i]
                inp = self.sigmoid(inp)
            return inp

    #--------------------------State functions--------------------------
    #   State prediction
    def pred_next_state(self, targ, dt):
        targ.pred_state = np.matmul(self.get_motion_matrix(targ.state, targ, dt), targ.state)
    
    #   State update - no detection
    def update_state_no_detection(self, targ, dt):
        targ.state = targ.pred_state

        targ.missed_detection = True
        targ.time_since_last_detection += dt

    #   State update - detection (detect -> [xc, yc, w, h])
    def update_state(self, targ, detect, dt):
        #Calculate error vector
        #   Calculate detection
        dpx, dpy, dpw, dph, ndt = 0, 0, 0, 0, 0
        if(targ.missed_detection): # TODO Make sure this is fine - here dt could be big and fuck up the taylor dev
            dpx = detect[0] - targ.last_detected_state[0,0]
            dpy = detect[1] - targ.last_detected_state[1,0]
            dpw = detect[2] - targ.last_detected_state[2,0]
            dph = detect[2] - targ.last_detected_state[2,0]
            ndt = targ.time_since_last_detection
            targ.missed_detection = False
            targ.time_since_last_detection = 0
        else:
            dpx = detect[0] - targ.state[0,0]
            dpy = detect[1] - targ.state[1,0]
            dpw = detect[2] - targ.state[2,0]
            dph = detect[3] - targ.state[3,0]
            ndt = dt
        a1 = (targ.last_grad[0][0]*targ.last_grad[0][3] - targ.last_grad[0][1]*targ.last_grad[0][2]) * ndt
        if(a1 < 1e-10): #TODO WTF is happening here...
            a1 = 1
        else:
            a1 = 1 / a1
        vx = (targ.last_grad[0][3] * dpx - targ.last_grad[0][1] * dpy) * a1
        vy = (targ.last_grad[0][0] * dpy - targ.last_grad[0][2] * dpx) * a1
        vw = dpw / ndt - (targ.last_grad[1][0] * vx + targ.last_grad[1][1] * vy)
        vh = dph / ndt - (targ.last_grad[1][2] * vx + targ.last_grad[1][3] * vy)
        detected_state = np.array([[detect[0], detect[1], detect[2], detect[3], vx, vy, vw, vh]]).T
    	#   Calculate error
        err = detected_state - targ.pred_state

        #Calculate interpolation vector
        inp = np.array([[err[0][0] / self.screen_w, err[1][0] / self.screen_h, 0 if targ.missed_detection else 1]])
        coeff = self.coeff_model.predict(inp)
        print(coeff)

        #Update state
        targ.state = targ.pred_state + coeff.T * err
        targ.last_detected_state = targ.state

        return err, inp

#Class to define the targets
class Atlante(object):
    def __init__(self, detection):
        #Init states
        self.state = np.array([[detection[0], detection[1], detection[2], detection[3], 0, 0, 0, 0]]).T
        self.pred_state = self.state

        #Init missed
        self.last_detected_state = self.state
        self.missed_detection = False
        self.time_since_last_detection = 0

        #Init other
        self.last_grad = None

if False:
    t = 0
    filt = Aquaman(300, 300, "aqua_weights_test.npy")
    filt.pre_calc(1)
    out = cv2.VideoWriter('gradient.avi', cv2.VideoWriter_fourcc(*'MP42'), float(24), (300, 300))
    while True:
        t += 0.1
        if t / (2*np.pi) > 1:
            break
        img = np.zeros((300,300,3), np.uint8)
        vx, vy = np.cos(t), np.sin(t)
        for x in range(0, 300, 1):
            for y in range(0, 300, 1):
                g = filt.get_gradient([x, y, 0, 0])[0]
                # img[x, y] = np.array([(g[0] * vx + g[1] * vy) * 255 / 2, 0, (g[2] * vx + g[3] * vy) * 255 / 2], np.uint8)
                v = np.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2] + g[3]*g[3]) / 4 * 255
                img[x, y] = np.array([v, v, v], np.uint8)
        out.write(img)
        cv2.imshow("Colors!!!!", img)
        print(t/(2*np.pi))

        k = cv2.waitKey(1)
        if k != -1:
            break
    out.release()
    cv2.destroyAllWindows()

f = 1
s = 600

def get_alpha(xr, yr): #xr, yr -> real world position at distance f
    d = np.sqrt(xr*xr + yr*yr)
    return np.abs(np.arctan(d/f))

def get_distorted_r(alpha):
    return f*np.sin(alpha)

def get_theta(xr, yr):
    t = np.arccos(xr/np.sqrt(xr*xr + yr*yr))
    return t if yr > 0 else 2*np.pi - t

def get_fish_box(xl, yl, xr, yr):
    a1 = get_alpha(xl, yl)
    a2 = get_alpha(xr, yr)
    d1 = get_distorted_r(a1)
    d2 = get_distorted_r(a2)
    t1 = get_theta(xl, yl)
    t2 = get_theta(xr, yr)
    
    nx1 = d1 * np.cos(t1) * s/2
    ny1 = d1 * np.sin(t1) * s/2
    nx2 = d2 * np.cos(t2) * s/2
    ny2 = d2 * np.sin(t2) * s/2

    return nx1 + s/2, ny1 + s/2, np.abs(nx1 - nx2), np.abs(ny1 - ny2)

if True:
    def draw_rect(img, state, color):
        cv2.rectangle(img, (int(state[0] - state[2]/2), int(state[1] - state[3]/2)), (int(state[0] + state[2]/2), int(state[1] + state[3]/2)), color, 2)
    theta = 0
    dt = 0.1
    real_state = np.array([0, 200, 10, 20])
    real_state = [*get_fish_box(real_state[0], real_state[1], real_state[0] + real_state[2], real_state[1] + real_state[3])]
    targ = Atlante(real_state)
    filt = Aquaman(600, 600, "aqua_weights_test.npy")
    filt.pre_calc(1)
    while True:
        t = time.perf_counter()
        theta += (2*np.pi/5 * dt) % (2 * np.pi)
        filt.pred_next_state(targ, 0.1)
        real_state = np.array([150 * np.sin(theta) + 300, 150 * np.sin(theta + np.pi / 4) + 300, 10, 20])
        #real_state = [*get_fish_box(real_state[0], real_state[1], real_state[0] + real_state[2], real_state[1] + real_state[3])]
        filt.update_state(targ, real_state, 0.1)

        print("dt : " + str(dt))
        print("--------------------")

        img = np.zeros((600, 600, 3), np.uint8)
        draw_rect(img, real_state, (255, 255, 255))
        draw_rect(img, targ.state.T[0], (0, 0, 255))
        cv2.imshow("Bob", img)
        k = cv2.waitKey(1)
        dt = time.perf_counter() - t
        if k == 27:
            break