import numpy as np

def normalize_angle(x, rad=True):
    if rad:
        return (x+np.pi)%(2*np.pi)-np.pi
    else:
        return (x+180.)%(360.)-180.

def score_frontier(tx, ty, yaw, fx, fy):
    """
    Rank Candidate Frontier Points.
    Currently, it chooses the one that is easiest to get to.
    (Minimum Distance & Angle)
    """

    dy = fy - ty
    dx = fx - tx

    # score 1 : distance from self ( better if small )
    fdist = np.sqrt(np.square(dy) + np.square(dx))

    # score 2 : angle from self : ( better if small )
    fang = np.arctan2(dy, dx) - yaw
    fang = (fang + np.pi)%(2*np.pi)-np.pi # normalize to +-pi
    fang = np.abs(fang)

    fscore = fdist * fang
    #fidx = np.argmin(fscore)
    fidx = np.argsort(fscore)
    return fx[fidx], fy[fidx]
    #return fx[fidx], fy[fidx]

