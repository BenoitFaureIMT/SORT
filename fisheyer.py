import numpy as np
import cv2

f = 1
distIm = 50

def get_alpha(xr, yr): #xr, yr -> real world position at distance f
    d = np.sqrt(xr*xr + yr*yr)
    return np.abs(np.arctan(d/distIm))

def get_distorted_r(alpha):
    return f*np.sin(alpha)

def get_theta(xr, yr):
    t = np.arccos(xr/np.sqrt(xr*xr + yr*yr))
    return t if yr > 0 else 2*np.pi - t

def get_xy(xr, yr):
    alpha = get_alpha(xr, yr)
    dist = get_distorted_r(alpha) / f
    theta = get_theta(xr, yr)
    return dist*np.cos(theta), dist*np.sin(theta)

def get_rthe(xr, yr):
    alpha = get_alpha(xr, yr)
    dist = get_distorted_r(alpha) / f
    theta = get_theta(xr, yr)
    return dist, theta

orImg = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
orImg = cv2.resize(orImg, (640, 640), cv2.INTER_LINEAR)

store = np.zeros((640, 640, 2))
minx, miny = 100000, 100000
maxx, maxy = -100000, -100000
for x in range(640):
    for y in range(640):
        nx, ny = (x - 320)/320, (y - 320)/320
        if not (nx ==0 and ny ==0):
            nx, ny = get_xy(nx, ny) #get_rthe(nx, ny)
        store[x, y] = [nx, ny]
        minx = min(nx, minx)
        miny = min(ny, miny)
        maxx = max(nx, maxx)
        maxy = max(ny, maxy)

for x in range(640):
    for y in range(640):
        nx, ny = store[x, y]
        orImg[x, y] = orImg[int((nx - minx)/(maxx - minx)*639), int((ny - miny)/(maxy - miny)*639)] #[int((nx - minx)/(maxx - minx)*255), int((ny - miny)/(maxy - miny)*255), 0]

cv2.imshow("fisheyer", orImg)
cv2.waitKey(0)