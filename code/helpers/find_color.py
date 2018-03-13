#!/usr/bin/python

import cv2
import numpy as np

scale = 2.0
f = '../../figures/rock.png'
img = cv2.imread(f)
img = cv2.resize(img, dsize=None, dst=img, fx=scale, fy=scale)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('win', img)

vs = []
def pxy(*args):
    k,x,y,t,_ = args
    if t == 1 and k==4:
        vs.append(hsv[y,x])
        lower = np.min(vs,axis=0)
        upper = np.min(vs,axis=0)
        print('limits : lower({}) ; upper({})'.format(lower, upper))

cv2.setMouseCallback('win', pxy)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()
