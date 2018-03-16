#!/usr/bin/python

import cv2
import argparse
import numpy as np

class FindColor(object):
    def __init__(self, scale, file):
        self._scale = 2.0
        self._file = file

        self._t_lo = np.uint8([0,0,0])
        self._t_hi = np.uint8([255,255,255])

        # Initialize UI
        cv2.namedWindow('img')
        cv2.createTrackbar('h_lo', 'img', 0, 255, (lambda v : self.set(v,0,True)))
        cv2.createTrackbar('s_lo', 'img', 0, 255, (lambda v : self.set(v,1,True)))
        cv2.createTrackbar('v_lo', 'img', 0, 255, (lambda v : self.set(v,2,True)))
        cv2.createTrackbar('h_hi', 'img', 0, 255, (lambda v : self.set(v,0,False)))
        cv2.createTrackbar('s_hi', 'img', 0, 255, (lambda v : self.set(v,1,False)))
        cv2.createTrackbar('v_hi', 'img', 0, 255, (lambda v : self.set(v,2,False)))
        cv2.setMouseCallback('img', self.click)

        # Initialize Images
        img = cv2.imread(self._file)
        img = cv2.resize(img, dsize=None, dst=img, fx=scale, fy=scale)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        trs = cv2.inRange(hsv, self._t_lo, self._t_hi)

        self._img = img
        self._hsv = hsv
        self._trs = trs
        self._update = False

    def set(self, value, index, low=True):
        if low:
            self._t_lo[index] = value
        else:
            self._t_hi[index] = value
        self._update = True

    def update(self):
        if self._update: # needs update
            self._update = False
            trs = cv2.inRange(self._hsv, self._t_lo, self._t_hi)
            self._trs=trs

    def show(self):
        cv2.imshow('img', self._img)
        cv2.imshow('hsv', self._hsv)
        cv2.imshow('trs', self._trs)

    def click(self, k, x, y, t, z):
        if k == cv2.EVENT_LBUTTONDOWN:
            print('HSV value @ ({},{}) : {}'.format(x, y, self._hsv[y,x]))

    def run(self):
        while True:
            k = cv2.waitKey(10)
            self.update()
            self.show()
            if k == 27:
                break

def main(args):
    app = FindColor(scale=args.scale, file=args.file)
    app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find Color Threshold.')
    parser.add_argument('file', type=str, help='image file to process')
    parser.add_argument('--scale', type=float, help='scaling parameter for the image', default=2.0)
    args = parser.parse_args()
    main(args)
