import cv2
import numpy as np

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

class ImageProcessor(object):
    def __init__(self, 
                 src, dst, scale,
                 th_nav, th_obs, th_roc,
                 th_deg, th_rng,
                 hsv=True
                ):
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._scale = scale
        self._th_nav = th_nav
        self._th_obs = th_obs
        self._th_roc = th_roc
        self._th_deg = th_deg
        self._th_rng = th_rng
        self._hsv = hsv # flag to convert to hsv
        
    def _cvt_rover(self, pi, pj, h, w):
        # pixel --> Rover
        px, py = float(h)-pi, float(w/2.0)-pj
        return px, py

    def _inv_rover(self, px, py, h, w):
        pi, pj = float(h)-px, float(w/2.0)-py
        return pi, pj

    def _cvt_polar(self, px, py):
        # Rover --> Polar
        r = np.sqrt(px**2+py**2)
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

    def range_filter(self, px, py, thresh):
        r, a = self._cvt_polar(px, py)
        thresh *= self._scale
        return px[r<=thresh], py[r<=thresh]

    def convert(self, img, thresh, yaw, tx, ty, mw, mh, polar=False, skip_thresh=False):
        h, w = img.shape[:2]
        if skip_thresh:
            sel = img
        else:
            sel = self._threshold(img, thresh)
        pi, pj = np.where(sel)
        px, py = self._cvt_rover(pi, pj, h, w)
        px, py = self.range_filter(px, py, self._th_rng)
        wx, wy = self._cvt_world(px, py, yaw, tx, ty, mw, mh)

        if polar:
            r, a = self._cvt_polar(px, py)
            return sel, (wx,wy), (r,a)

        return sel, (wx,wy)
        
    def __call__(self, rover):
        """ Assume RGB Image Input """

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

        #bound = np.logical_and(map[:,:,0], map[:,:,2])
        bound = map[:,:,0]
        cv2.imshow('map', np.flipud(map[:,:,::-1]))
        #cv2.imshow('bound', np.flipud(np.float32(bound)))
        cv2.waitKey(10)

        # Warp
        warped = cv2.warpPerspective(img, self._M, (w,h))# keep same size as input image
        if self._hsv:
            warped = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
            
        # Threshold
        nav, (wx, wy), (r,a) = self.convert(warped, self._th_nav, yaw, tx, ty, mw, mh, polar=True)
        if update_map:
            map[wy, wx, 2] = np.clip(map[wy,wx,2]+1, 0, 255)
            map[wy, wx, 0] = np.clip(map[wy,wx,0]-1, 0, 255)

        obs, (wx, wy) = self.convert(warped, self._th_obs, yaw, tx, ty, mw, mh, polar=False)

        # closest-obstacle match
        #img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #obs = self._threshold(img_hsv, self._th_obs) 

        #obsb = last_nonzero(obs, axis=0) 
        #p_obs = np.stack((np.arange(len(obsb)), obsb), axis=1) #(j,i)

        #obs.fill(0)
        #obs[p_obs[:,1], p_obs[:,0]] = 1
        #print obs.dtype
        #obs = cv2.warpPerspective(np.uint8(obs), self._M, (w,h)).astype(np.bool)

        #p_obs = p_obs[p_obs[:,1] > 60]
        #p_obs_f = np.float32(p_obs)
        #pi, pj = ptrans(p_obs_f, self._M, w, h).T
        #oviz = np.zeros_like(warped)
        #oviz[pi, pj] = 255
        #oviz = cv2.dilate(oviz, np.ones((5,5)), iterations=1)

        #px, py = self._cvt_rover(pi, pj, h, w)
        #px, py = self.range_filter(px, py)
        #wx, wy = self._cvt_world(px, py, yaw, tx, ty, mw, mh)

        #print np.shape(k)
        #pj, pi = pi[pi!=0], pj[pi!=0]
        #px,py = self._cvt_rover(pi, pj, h, w)
        #wx,wy = self._cvt_world(px, py, yaw, tx, ty, mw, mh)
        #obs, (wx,wy) = self.convert(warped, self._th_obs, yaw, tx, ty, mw, mh, polar=False)


        if update_map:
            #mask = np.zeros((h,w), dtype=np.uint8)
            #mask[obs] = 255
            #im2, cntr, hrch = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #mask.fill(0)
            #cv2.drawContours(mask, cntr, -1, 255, 3)
            #mask = mask.astype(np.bool)
            #_, (wx,wy) = self.convert(mask, self._th_obs, yaw, tx, ty, mw, mh, polar=False, skip_thresh=True)

            map[wy, wx, 0] = np.clip(map[wy,wx,0]+1, 0, 255)
            map[wy, wx, 2] = np.clip(map[wy,wx,2]-1, 0, 255)

        stat = np.zeros_like(warped)
        stat[nav,2] = 255
        stat[:, 1 ] = 0
        stat[obs,0] = 255
        #stat[pi,pj,0] = 255

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
            th_deg = 3.0,
            th_rng = 6.0,
            hsv = True
            )
    m = proc.get_range_mask()
    cv2.imshow('mask', np.float32(m))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
