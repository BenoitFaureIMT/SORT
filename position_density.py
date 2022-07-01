from tkinter import N
import cv2
import numpy as np

f = 1

def get_alpha(xr, yr): #xr, yr -> real world position at distance f
    d = np.sqrt(xr*xr + yr*yr)
    return np.abs(np.arctan(d/f))

def get_distorted_r(alpha):
    return f*np.sin(alpha)

def get_theta(xr, yr):
    t = np.arccos(xr/np.sqrt(xr*xr + yr*yr))
    return t if yr > 0 else 2*np.pi - t

img = np.zeros((640, 640, 3), np.uint8)
img2 = np.zeros((640, 640, 3), np.uint8)

for x in range(1000):
    cv2.circle(img2, (320, 320), int(get_distorted_r(get_alpha(x/10, 0))/f*320), (255, 255, 255), 1)

for rr in range(100):
    nr = rr/10
    for xr in range(-int(nr * 100), int(nr * 100)):
        nxr = xr / 100

        yr = np.sqrt(nr*nr - (nxr)**2)
        r = get_distorted_r(get_alpha(nxr, yr))  / f * 320
        theta = get_theta(nxr, yr)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        cv2.circle(img, (int(x) + 320, int(y) + 320), 1, (255, 255, 255), 1)

        yrm = -yr
        rm = get_distorted_r(get_alpha(nxr, yrm))  / f * 320
        thetam = get_theta(nxr, yrm)
        xm = rm * np.cos(thetam)
        ym = rm * np.sin(thetam)
        cv2.circle(img, (int(xm) + 320, int(ym) + 320), 1, (255, 255, 255), 1)


cv2.imshow('Point distortion',img)
cv2.imshow('Circle distortion',img2)
cv2.waitKey(0)