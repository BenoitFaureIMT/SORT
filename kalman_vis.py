from calendar import c
from re import I
from tkinter import N
import cv2
import numpy as np
from target_util import target

f = 1
s = 640

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
    
    nx1 = d1 * np.cos(t1)
    ny1 = d1 * np.sin(t1)
    nx2 = d2 * np.cos(t2)
    ny2 = d2 * np.sin(t2)

    return nx1, ny1, np.abs(nx1 - nx2), np.abs(ny1 - ny2)

def draw_box(x, y, w, h, img, col = (255, 255, 255)):
    cv2.rectangle(img, (int(x + 320), int(y + 320)), (int(x + w + 320), int(y + h + 320)), col, 1)

#bbox
# xl,yl,xr,yr = -100, -100, -50, 0

# img = np.zeros((640, 640, 3), np.uint8)
# x,y,w,h = get_fish_box(xl, yl, xr, yr)
# print(x, y, w, h)
# x, y, w, h = (x*320, y*320, w*320, h*320)
# draw_box(x, y, w, h, img)
# draw_box(xl, yl, xr-xl, yr-yl, img)

vid = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'MP42'), float(24), (640, 640))

#bbox
xl,yl,xr,yr = -.5, -.5, -.25, 0
x,y,w,h = get_fish_box(xl, yl, xr, yr)
targ = target([x, y, w, h, 1], 1/24)

img = np.zeros((640, 640, 3), np.uint8)
x, y, w, h = (x*s, y*s, w*s, h*s)
draw_box(x, y, w, h, img, 1)
vid.write(img)

for fr in range(1, 480):
    xl += np.sin(fr/240 * np.pi)/200
    xr += np.sin(fr/240 * np.pi)/200
    yl += np.sin(fr/240 * 2 * np.pi)/200
    yr += np.sin(fr/240 * 2 * np.pi)/200

    c = np.random.random()*0.4 + 0.6
    x,y,w,h = get_fish_box(xl, yl, xr, yr)
    targ.update_pred()
    targ.update_state([x, y, w, h, c])

    img = np.zeros((640, 640, 3), np.uint8)
    x, y, w, h = (x*s, y*s, w*s, h*s)
    draw_box(x, y, w, h, img)
    dr =  [v * s for v in targ.center_to_corner(targ.state)]
    draw_box(dr[0], dr[1], dr[2], dr[3], img, col = (255, 0, 0))
    vid.write(img)

vid.release()



# cv2.imshow("Actual position", img)
# cv2.waitKey(0)