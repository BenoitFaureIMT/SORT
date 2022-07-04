import numpy as np
import matplotlib.pyplot as plt
import cv2

orImg = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
orImg = cv2.resize(orImg, (640, 640), cv2.INTER_LINEAR)

img = np.zeros((640, 640, 3))
z = 0.5

store = np.zeros((640, 640))
mr = -10000000
for x in range(640):
    for y in range(640):
        nx, ny = float(x - 320)/320, float(y - 320)/320
        r = nx*nx + ny*ny
        if r < 1:
            r = np.sqrt(r) + (1 - np.sqrt(1 - r))/2
        else:
            r = 0
        store[x, y] = r
        mr = max(r, mr)

for x in range(640):
    for y in range(640):
        nx, ny = (x - 320), (y - 320)
        a = np.sqrt(nx*nx + ny*ny)
        if a > 0.001:
            nx /= a
            ny /= a
        
        nr = store[x,y] * z / mr * 320
        ox, oy = (int(nr * nx + 320), int(nr * ny + 320))
        ox, oy = min(max(ox, 0), 639), min(max(oy, 0), 639)
        img[x, y] = orImg[ox, oy] / 255

        # v = store[x,y] / mr
        # img[x, y] = [v, v, v]

cv2.imshow("fisheyer v2", img)
cv2.imshow("fisheyer v2 - real", orImg)
cv2.waitKey(0)