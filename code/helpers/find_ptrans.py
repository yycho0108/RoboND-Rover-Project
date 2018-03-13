import numpy as np
import cv2

def ptrans(pts, M, w, h):
    n = len(pts)
    pts = np.concatenate((pts, np.ones((n,1))), axis=-1)
    
    pp = M.dot(pts.T)
    pp[:2] /= pp[2]
    res = pp[:2].T

    a = np.logical_and
    res = np.int32(np.round(res))

    #i_good = a(
    #        a(0 <= res[:,0], res[:,0] < w),
    #        a(0 <= res[:,1], res[:,1] < h)
    #        )
    return res[:, ::-1]
    #return res[i_good, ::-1]

N = 1000
H = 160
W = 320
M = np.float32([
        [-2.66018968e-02, -1.97837150e+00,  1.64086283e+02],
        [ 8.65973959e-15, -1.98924358e+00,  1.63662040e+02],
        [ 3.81639165e-17, -1.23756650e-02,  1.00000000e+00]])

src = np.zeros((H,W), dtype=np.float32)

k = 2
src[H/5,:] = 1
src[:,W/5] = 1
src[(k-1)*H/k,:] = 1
src[:,(k-1)*W/k] = 1
#src = np.ones_like(src)
i,j = np.where(src)
#i = np.random.randint(low=0, high=H, size=N)
#j = np.random.randint(low=0, high=W, size=N)
srcv = np.stack([j, i], axis=1)# as vector
src[i,j] = 1.0

#srcv = np.expand_dims(srcv, axis=0).astype(np.float32)
#dst1 = cv2.perspectiveTransform(srcv, M)[0]
dst1 = ptrans(srcv, M, W, H)
dst1_viz = np.zeros_like(src)
print np.min(dst1, axis=0)
print np.max(dst1, axis=0)
print np.shape(dst1_viz)
print np.shape(dst1)
dst1_viz[dst1[:,0], dst1[:,1]] = 1.0

dst2 = cv2.warpPerspective(src, M, (W,H)).nonzero()
dst2_viz = np.zeros_like(src)
dst2_viz[dst2] = 1.0

cv2.imshow('d1', dst1_viz)
cv2.imshow('d2', dst2_viz)

while True:
    if cv2.waitKey(20) == 27:
        break
