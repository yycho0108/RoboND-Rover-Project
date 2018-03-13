import cv2
import numpy as np
from astar import EDist, AStar2DGrid as AStar
import sys

from utils import normalize_angle


def cproc(cnt):
    """
    For simple closed-contours that are 
    omposed of nearly parallel lines,
    cproc() will roll the contour such tha
    its geometric extremes will be located at the ends.
    """
    cnt = np.squeeze(cnt, axis=1)
    cnt = np.roll(cnt, np.random.randint(256), axis=0)

    n = len(cnt)
    if n <= 2:
        return cnt
    n2 = n/2 # 1
    c1 = cnt[:n2]
    c2 = cnt[-n2:]
    #c2 = np.flip(cnt[-n2:], axis=0)
    diffs = np.linalg.norm(np.subtract(c1,c2), axis=-1)
    #print diffs
    m = np.argmax(diffs)

    cnt = np.roll(cnt, -m, axis=0)
    c1 = cnt[:n2]
    return c1
    #c2 = np.flip(cnt[-n2:], axis=0)
    #return np.mean([c1,c2], axis=0).astype(np.int32)

class ImageProcessor(object):
    """
    Bulk of Image-processing pipeline.
    Contains coordinate-conversion utility functions,
    as well as most of image-processing functions used in the project.
    """
    def __init__(self, 
                 src, dst, scale,
                 th_nav, th_obs, th_roc,
                 th_deg, th_rng, th_ang,
                 hsv=True
                ):
        """
        Parameters:
            src, dst : defines points for perspective transformation.
            th_nav : Color threshold for Navigable terrain; hsv if hsv=True.
            th_obs : Color threshold for Obstacle terrain; hsv if hsv=True.
            th_roc : Color threshold for Rocks; hsv if hsv=True.
            th_deg : Yaw/Roll threshold for updating maps.
            th_rng : View Range threshold for updating maps.
            th_ang : View Angle threshold for updating maps
            hsv : Flag; whether or not color coordinates are hsv. default=True
        """
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._scale = scale
        self._th_nav = th_nav
        self._th_obs = th_obs
        self._th_roc = th_roc
        self._th_deg = th_deg
        self._th_rng = th_rng
        self._th_ang = th_ang
        self._hsv = hsv # flag to convert to hsv
        self._first = True
        
    def _cvt_rover(self, pi, pj, h, w):
        """ Pixel -> Rover """
        px, py = float(h)-pi, float(w/2.0)-pj
        return px, py

    def _inv_rover(self, px, py, h, w):
        """ Rover -> Pixel """
        pi, pj = float(h)-px, float(w/2.0)-py
        return pi, pj

    def _cvt_polar(self, px, py):
        """ Rover -> Polar """
        r = np.sqrt(np.square(px)+np.square(py))
        r /= self._scale
        h = np.arctan2(py,px)
        return r, h

    def _inv_polar(self, r, h):
        """ Polar -> Rover """
        x = r * np.cos(h)
        y = r * np.sin(h)
        return x, y
    
    def _cvt_world(self, px, py, yaw, tx, ty, mw, mh):
        """ Rover -> World """
        #yaw = np.deg2rad(yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        rmat = np.reshape([c,-s,s,c], (2,2))
        wx, wy = np.int32(np.add(np.dot(rmat, [px,py])/self._scale, [[tx],[ty]]))
        
        wx = np.clip(wx, 0, mw-1)
        wy = np.clip(wy, 0, mh-1)
        return wx, wy

    def _threshold(self, img, thresh):
        """
        Applies color threshold to img.
        Parameters:
            img : image to apply color transform
            thresh : (low, high) with each same depth as image.
        """
        sel = cv2.inRange(img, thresh[0], thresh[1]).astype(np.bool)
        return sel

    def range_filter(self, px, py):
        """
        Filter observations by range to limit junk information.
        """
        r, a = self._cvt_polar(px, py)
        aa = np.abs(a)
        good = np.logical_and(r<=self._th_rng, aa<=self._th_ang)
        return px[good], py[good]

    def convert(self, img, thresh, yaw, tx, ty, mw, mh, polar=False, skip_thresh=False):
        """
        Apply most of necessary conversions automatically.
        """
        h, w = img.shape[:2]
        if skip_thresh:
            sel = img
        else:
            sel = self._threshold(img, thresh)
        pi, pj = np.where(sel)
        px, py = self._cvt_rover(pi, pj, h, w)
        px, py = self.range_filter(px, py)
        wx, wy = self._cvt_world(px, py, yaw, tx, ty, mw, mh)

        if polar:
            r, a = self._cvt_polar(px, py)
            return sel, (wx,wy), (r,a)

        return sel, (wx,wy)
       
    def __call__(self, rover):
        """ Assume RGB Image Input """

        if self._first:
            # initialize.
            # TODO : fix more properly
            self._first = False
            rover.goal = None
            rover.p0 = None
            rover.path = None
            rover.rock = None

        # Unpack Data
        img = rover.img
        tx, ty = rover.pos
        yaw, pitch, roll = [np.deg2rad(e) for e in rover.yaw, rover.pitch, rover.roll]
        yaw, pitch, roll = [normalize_angle(e) for e in [yaw,pitch,roll]] #to +-np.pi
        map, map_gt = rover.worldmap, rover.ground_truth

        # decide whether or not data is good in general
        update_map = abs(pitch) < np.deg2rad(self._th_deg) and \
                abs(roll) < np.deg2rad(self._th_deg)

        h, w = img.shape[:2]
        mh, mw = map.shape[:2]
        #cv2.imshow('mapped', np.flipud(mapped))

        # Warp
        warped = cv2.warpPerspective(img, self._M, (w,h))# keep same size as input image
        if self._hsv:
            warped = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
            
        # Threshold
        nav, (wx, wy), (r,a) = self.convert(warped, self._th_nav, yaw, tx, ty, mw, mh, polar=True)
        if update_map:
            map[wy, wx, 2] = np.clip(map[wy,wx,2]+10, 0, 255)
            map[wy, wx, 0] = np.clip(map[wy,wx,0]-10, 0, 255)

        obs, (wx, wy) = self.convert(warped, self._th_obs, yaw, tx, ty, mw, mh, polar=False)
        if update_map:
            map[wy, wx, 0] = np.clip(map[wy,wx,0]+5, 0, 255)
            map[wy, wx, 2] = np.clip(map[wy,wx,2]-5, 0, 255)

        roc, (wx, wy) = self.convert(warped, self._th_roc, yaw, tx, ty, mw, mh, polar=False)
        if update_map:
            map[wy, wx, 1] = np.clip(map[wy,wx,1]+1, 0, 255)

        if len(wx) > 0:
            rover.rock = (np.mean(wx), np.mean(wy))
        else:
            rover.rock = None
            #

        stat = np.zeros_like(warped)
        stat[nav,2] = 255
        stat[:, 1 ] = 0
        stat[obs,0] = 255

        for rad in range(10):
            cv2.circle(stat, (w/2,h), rad*10, (255,255,255), 1)

        thresh = np.zeros_like(img)
        thresh[:, :, 2] = 255*self._threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
                self._th_nav
                )

        thresh[:, :, 0] = 255*self._threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
                self._th_obs
                )

        # Create Visualizations
        overlay = cv2.addWeighted(map, 1, map_gt, 0.5, 0)
        maxh = max(h,mh)
        maxw = max(w,mw)
        viz = np.zeros((maxh*2, maxw*2, 3))
        viz[0:h, 0:w] = img
        #viz[0:h, w:w+w] = warped
        viz[0:h, w:w+w] = stat
        viz[h:h+mh, 0:mw] = np.flipud(overlay)
        #viz[h:h+mh, mw:mw+mw] = np.flipud(map)
        viz[h:h+h, mw:mw+w] = thresh
        
        return viz, (r,a)

    def get_range_mask(self):
        h,w = 160,320
        sel = np.ones((h,w), dtype=np.bool)
        pi, pj = np.where(sel)
        px, py = self._cvt_rover(pi, pj, h, w)
        r, a = self._cvt_polar(px, py)
        #print np.min(a), np.max(a)
        px, py = self.range_filter(px, py)
        pi, pj = np.int32(self._inv_rover(px, py, h, w))
        sel[pi, pj] = 0
        sel = np.logical_not(sel)
        return sel

def main():
    h,w = 160, 320
    scale = 10
    s2 = scale/2
    bottom_offset = 6
    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[w/2 - s2, h - bottom_offset],
                      [w/2 + s2, h - bottom_offset],
                      [w/2 + s2, h - 2*s2- bottom_offset], 
                      [w/2 - s2, h - 2*s2- bottom_offset],
                      ])

    proc = ImageProcessor(
            src=src,
            dst=dst,
            scale=scale,
            th_nav = ((0,0,180), (50,50,256)),
            th_obs = ((0,0,1), (35,256,90)),
            th_roc = ((0,0,0), (255,255,256)), # don't know yet
            th_deg = 0.5,
            th_rng = 6.0,
            th_ang = np.deg2rad(75),
            hsv = True
            )
    m = proc.get_range_mask()
    cv2.imshow('mask', np.float32(m))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
