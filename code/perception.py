import numpy as np
import cv2
from img_proc import ImageProcessor

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly

#src = np.float32([
#    [120, 98],
#    [200, 98],
#    [310, 144],
#    [16, 144], 
#    ])
#    
#dst = 0.5*np.float32([
#    [-1, -1], 
#    [1, -1], 
#    [1, 1],
#    [-1, 1], 
#    ])
#offset = 3
#dst = np.add(scale*dst, (w/2.0,h-scale/2-offset)).astype(np.float32)


proc = None
def perception_step(Rover):
    global proc
    if proc is None:
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

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # proc automatically updates Rover.worldmap
    viz, (r,a) = proc(Rover)
    # r,a = polar-coord representation of navigable terrain
    Rover.vision_image = viz
    Rover.nav_dists = r
    Rover.nav_angles = a
    return Rover
