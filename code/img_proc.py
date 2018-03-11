import cv2
import numpy as np
from astar import EDist, AStar2DGrid as AStar
import sys

def last_nonzero(mask, axis, invalid_val=-1):
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def ptrans(pts, M, w=0, h=0):
    n = len(pts)
    pts = np.concatenate((pts, np.ones((n,1))), axis=-1)
    
    pp = M.dot(pts.T)
    pp[:2] /= pp[2]
    res = pp[:2].T

    a = np.logical_and
    res = np.int32(np.round(res))

    if w>0 and h>0:
        i_good = a(
                a(0 <= res[:,0], res[:,0] < w),
                a(0 <= res[:,1], res[:,1] < h)
                )
        return res[i_good, ::-1]
    return res[:, ::-1]

def skeleton(img, ker=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))):
    size = np.size(img)
    skel = np.zeros_like(img, dtype=np.uint8)
    while True:
        eroded = cv2.erode(img, ker)
        temp = cv2.dilate(eroded ,ker)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            return skel

def score_frontier(tx, ty, yaw, fx, fy):
    dy = fy - ty
    dx = fx - tx

    # score 1 : distance from self ( better if small )
    fdist = np.sqrt(np.square(dy) + np.square(dx))

    # score 2 : angle from self : ( better if small )
    fang = np.arctan2(dy, dx) - yaw
    fang = (fang + np.pi)%(2*np.pi)-np.pi # normalize to +-pi
    fang = np.abs(fang)

    fscore = fdist * fang
    fidx = np.argmin(fscore)

    return fx[fidx], fy[fidx]


def cproc(cnt):
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
    def __init__(self, 
                 src, dst, scale,
                 th_nav, th_obs, th_roc,
                 th_deg, th_rng, th_ang,
                 hsv=True
                ):
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
        # pixel --> Rover
        px, py = float(h)-pi, float(w/2.0)-pj
        return px, py

    def _inv_rover(self, px, py, h, w):
        pi, pj = float(h)-px, float(w/2.0)-py
        return pi, pj

    def _cvt_polar(self, px, py):
        # Rover --> Polar
        r = np.sqrt(np.square(px)+np.square(py))
        r /= self._scale
        h = np.arctan2(py,px)
        return r, h

    def _inv_polar(self, r, h):
        x = r * np.cos(h)
        y = r * np.sin(h)
        return x, y
    
    def _cvt_world(self, px, py, yaw, tx, ty, mw, mh):
        # Rover --> World
        #yaw = np.deg2rad(yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        rmat = np.reshape([c,-s,s,c], (2,2))
        wx, wy = np.int32(np.add(np.dot(rmat, [px,py])/self._scale, [[tx],[ty]]))
        
        wx = np.clip(wx, 0, mw-1)
        wy = np.clip(wy, 0, mh-1)
        return wx, wy

    def _threshold(self, img, thresh):
        #sel = np.logical_and(
        #        np.sum(thresh[0]<=img, axis=2)==3,
        #        np.sum(img<thresh[1],  axis=2)==3
        #        )
        ##return sel
        #print 'ss0', sel.shape
        #print 'ssd', sel.dtype
        sel = cv2.inRange(img, thresh[0], thresh[1]).astype(np.bool)
        return sel

    def range_filter(self, px, py):
        r, a = self._cvt_polar(px, py)
        aa = np.abs(a)
        good = np.logical_and(r<=self._th_rng, aa<=self._th_ang)
        return px[good], py[good]

    def convert(self, img, thresh, yaw, tx, ty, mw, mh, polar=False, skip_thresh=False):
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

    def find_goal(self, segs):
        #map_obs = map[:,:,0]
        #map_nav = map[:,:,2]
        #_, cnt, _ = cv2.findContours(map_nav.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #_, cnt, _ = cv2.findContours(bound.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnt  = [cproc(c) for c in cnt]
        # filter contours by length
        #cnt = [c for c in cnt if cv2.arcLength(c, False) > 5.0]
        cnt = segs

        # find best endpoint to explore
        m = len(cnt)
        goals = []
        goal = None

        if m == 1:
            # only one contour
            i=j=0
            goal = (cnt[0][0] + cnt[0][-1]) / 2.0
            goals.append(goal)
        else:
            for i in range(m):
                mn = 30.0 #3m
                for j in range(i+1,m):
                    ci = np.expand_dims([cnt[i][0], cnt[i][-1]], 1) # 2, 1, 2
                    cj = np.expand_dims([cnt[j][0], cnt[j][-1]], 0) # 1, 2, 2
                    dc = np.linalg.norm(ci-cj, axis=-1) # 2, 2, 2 -> 2,2
                    mi, mj = np.unravel_index(dc.argmin(), dc.shape)
                    if dc[mi,mj] < mn:
                        mn = dc[mi,mj]
                        goal = (ci[mi,0] + cj[0,mj]) / 2.0
                        goals.append(goal)

        
        return cnt, goals
        
    def __call__(self, rover):
        """ Assume RGB Image Input """

        if self._first:
            self._first = False
            rover.goal = None
            rover.p0 = None
            rover.next_goal = None
            rover.path = None

        # Unpack Data
        img = rover.img
        tx, ty = rover.pos
        yaw, pitch, roll = [np.deg2rad(e) for e in rover.yaw, rover.pitch, rover.roll]
        if pitch > np.pi:
            pitch -= 2*np.pi
        if roll > np.pi:
            roll -= 2*np.pi
        map, map_gt = rover.worldmap, rover.ground_truth

        update_map = abs(pitch) < np.deg2rad(self._th_deg) and \
                abs(roll) < np.deg2rad(self._th_deg)

        h, w = img.shape[:2]
        mh, mw = map.shape[:2]

        # Frontier
        #map_obs = map[:,:,0]
        #map_nav = cv2.dilate(map[:,:,2], np.ones([5,5]), iterations=1)
        #map_nav_out = cv2.dilate(map[:,:,2], np.ones([3,3]), iterations=1)
        #bound = np.logical_and(map_nav_out, map_obs)

        map_nav = map[:,:,2]
        map_obs = map[:,:,0]
        ker = cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3))

        mapped = np.logical_or(
                np.greater(map_nav, 20),
                np.greater(map_obs, 2),
                )
        mapped = 255 * mapped.astype(np.uint8)
        mapped = cv2.erode(mapped, cv2.getStructuringElement(cv2.MORPH_ERODE, (3,3)), iterations=1)

        cv2.imshow('mapped', np.flipud(mapped))

        cnt = cv2.findContours(mapped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        mapped.fill(0)
        goal = None
        if len(cnt) > 0:
            map_nav = cv2.dilate(map_nav, ker, iterations=1)
            cv2.drawContours(mapped, cnt, -1, 255)
            frontier = np.logical_and(map_nav, mapped)
            frontier = 255 * frontier.astype(np.uint8)
            
            fy, fx = frontier.nonzero() #(2,N)

            # basic filter : no obstacles!
            good_goal = (map_obs[fy,fx] <= 1)
            fy = fy[good_goal]
            fx = fx[good_goal]

            if np.size(fy) > 0:
                goal = score_frontier(tx, ty, yaw, fx, fy)

            #dy = fy - ty
            #dx = fx - tx
            #fdist = np.sqrt(np.square(fy-ty) + np.square(fx-tx))

            #if np.size(fdist) > 0:
            #    fidx = np.argmin(fdist)
            #    #print 'cur {} -> nxt {}'.format( (tx,ty), (fx[fidx], fy[fidx]) )
            #    goal = (fx[fidx], fy[fidx])

            #cv2.imshow('frontier', np.flipud(frontier))

        #_, cnt, _ = cv2.findContours(
        #        (255* np.greater(map_nav, 20)).astype(np.uint8),
        #        cv2.RETR_EXTERNAL,
        #        cv2.CHAIN_APPROX_SIMPLE)

        #map_nav = cv2.dilate(map_nav, ker, iterations=1)
        #map_obs = cv2.dilate(map_obs, ker, iterations=1)

        #segs = []
        #if len(cnt) > 0:
        #    c = cnt[0][:,0] # --> (N,2) 
        #    seg = []
        #    for pt in c:
        #        if not map_obs[pt[1], pt[0]] > 10:
        #            if seg:
        #                segs.append(np.int32(seg))
        #            seg = []
        #        else:
        #            seg.append([pt[0], pt[1]])

        #map_cnt = np.zeros_like(map_obs, dtype=np.uint8)
        #cv2.drawContours(map_cnt, cnt, -1, 255)
        #np.logical_and(map_cnt, np.greater(map_obs, 20), map_cnt)
        #cv2.imshow('mcnt', np.float32(map_cnt))
        #_, cnt, _ = cv2.findContours(
        #        (255* np.greater(map_cnt, 20)).astype(np.uint8),
        #        cv2.RETR_EXTERNAL,
        #        cv2.CHAIN_APPROX_SIMPLE)


        bound = (np.logical_and(
            np.greater(map_nav, 20),
            np.greater(map_obs, 20)) * 255).astype(np.uint8)
        bound = cv2.cvtColor(bound, cv2.COLOR_GRAY2BGR)
        #for i in range(len(cnt)):
        #    cv2.drawContours(bound, cnt, i, np.random.randint(255, size=3), 1)

        rover.next_goal = goal

        # visualization ...
        #cimg = np.zeros((mh,mw,3), dtype=np.float32)
        cimg = np.copy(bound)
        if rover.goal is not None:
            cv2.circle(cimg, tuple(np.int_(rover.pos)), 2, [0.0, 255, 0.0])
            cv2.circle(cimg, tuple(np.int_(rover.goal)), 2, [0.0,0.0,255])
        if rover.path is not None:
            for (p0, p1) in zip(rover.path[:-1], rover.path[1:]):
                x0,y0 = p0
                x1,y1 = p1
                #cv2.line( (y0,x0), (y1,x1), (128)
                cv2.line(cimg, (x0,y0), (x1,y1), (255,0,0), 1)

        if rover.p0 is not None:
            for (p0, p1) in zip(rover.p0[:-1], rover.p0[1:]):
                x0,y0 = p0
                x1,y1 = p1
                #cv2.line( (y0,x0), (y1,x1), (128)
                cv2.line(cimg, (x0,y0), (x1,y1), (255,255,0), 1)

        #for i, c in enumerate(segs):
        #    #print np.shape(c)
        #    cv2.polylines(cimg, c[np.newaxis, ...], False, color=np.random.uniform(size=3), thickness=1)

        #cv2.imshow('bound', np.flipud(bound))
        cv2.imshow('cimg', np.flipud(cimg))
        cv2.waitKey(10)

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

        stat = np.zeros_like(warped)
        stat[nav,2] = 255
        stat[:, 1 ] = 0
        stat[obs,0] = 255

        #rover.local_map = self._cvt_rover(

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
