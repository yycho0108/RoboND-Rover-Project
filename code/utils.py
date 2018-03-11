import numpy as np

def normalize_angle(x, rad=True):
    if rad:
        return (x+np.pi)%(2*np.pi)-np.pi
    else:
        return (x+180.)%(360.)-180.
