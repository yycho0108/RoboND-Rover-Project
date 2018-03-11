import numpy as np
import cv2
from img_proc import ImageProcessor

"""
Default implementations have been moved to ImageProcessor() @ img_proc.py
In favor of trying to organize everything in order.
"""

proc = None
def perception_step(Rover):
    global proc
    if proc is None:
        # initialize processor
        h, w = np.shape(Rover.img)[:2]
        scale = 10.
        dst_size = scale / 2
        bottom_offset = 6
        src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
        dst = np.float32([[w/2 - dst_size, h - bottom_offset],
                          [w/2 + dst_size, h - bottom_offset],
                          [w/2 + dst_size, h - 2*dst_size - bottom_offset], 
                          [w/2 - dst_size, h - 2*dst_size - bottom_offset],
                          ])

        proc = ImageProcessor(
                src=src,
                dst=dst,
                scale=scale,
                th_nav = ((0,0,180), (50,50,256)),
                th_obs = ((0,0,1), (35,256,90)),
                th_roc = ((20,230,100), (30,256,230)), # don't know yet
                th_deg = 1.0,
                th_rng = 8.0,
                th_ang = np.deg2rad(75),
                hsv=True
                )

    # Note that ImageProcessor() also modifies Rover's internal states.
    viz, (r,a) = proc(Rover)

    Rover.vision_image = viz
    Rover.nav_dists = r
    Rover.nav_angles = a
    return Rover
