# Project : Search and Sample Return

Yoonyoung Cho | 03/20/2018

# Description

This project is modeled after the [NASA sample return challenge](https://www.nasa.gov/directorates/spacetech/centennial_challenges/sample_return_robot/index.html), and serves as the first project in the Udacity Robotics Software Nanodegree Term 1.

The objective is to implement an autonomous ground-rover that explores through an unknown environment in search of rock samples. The rover is given some prior knowledge of the environment, such as general visual features, but without access to full information such as the terrain map or the sample locations. The lower-level controls, such as precise actuation, have been abstracted so that the focus is on high-level planning and visual perception of the environment through a frontal camera.

See [here](./README.md) for the complete formal description from Udacity.

# Notebook Analysis

In the beginning phase of the project, I implemented the visual-processing pipeline in a Jupyter Notebook, with use of recorded data.

See the [notebook](./code/Rover_Project_Test_Notebook.ipynb) for reference throughout this section.

## Color Selection

The first step in the image-processing pipeline was color selection, in order to identify objects based on their color features. Given the relatively simple visual environment, this was an effective sensing strategy.

Unlike the examples provided, I first converted the input image in the HSV colorspace in order to minimize sensitivity to brightness, or saturation ([image source](https://www.researchgate.net/figure/HSV-Color-Space_fig2_284488273)).

![HSV](./figures/HSV-Color-Space-Downsize.jpg)

The three types of environmental markers (ground plane, obstacles and rocks) were clearly distinguishable in the RGB colorspace as well, so it was mostly a matter of personal preference. I wanted to take extra precaution in order to avoid confusion between by the presence of shadows and lit points on the ground (even though the difference was mostly negligible), which may have influenced the perception of the rover throughout the mission at unexpected points.

In order to effectively identify the thresholds, I created a [separate script](./code/helpers/find_color.py) that took in an image and basically allowed the user to find lower and upper bounds by clicking on multiple sample points (for reference), and adjusting trackbars to apply the bounds on the hsv image to produce the thresholded image. I use this functionality often enough, that I wanted to write something nice once, that I could use multiple times in the future. Here's what a typical interaction would look like:

![find\_color](./figures/find_color.png)

After running through the script, I identified the following lower and upper bounds of the objects in HSV colorspace
(bear in mind that, in OpenCV, the default range for hue is from 0-180, scaled down by a factor of 2):

```python
th_nav = ((0,0,180), (50,50,256)),
th_obs = ((0,0,1), (35,256,90)),
th_roc = ((20,230,100), (30,256,230))
```

Refer to [perception.py](./code/perception.py) for where this information can be found.

These thresholds were then applied in [img\_proc.py](./code/img_proc.py) as a class method under `ImageProcessor()`:

```python
def _threshold(self, img, thresh):
    """
    Applies color threshold to img.
    Parameters:
        img : image to apply color transform
        thresh : (low, high) with each same depth and colorspace as image.
    """
    sel = cv2.inRange(img, thresh[0], thresh[1]).astype(np.bool)
    return sel
```

I simply called opencv's native function rather than chaining numpy's comparison and logical operations, in order to have a more compact and efficient expression.

### Visualization

In perspective, the following diagram spatially represents the previously defined color thresholds (the diagram can be reproduced through [col\_viz.py](./code/col_viz.py)):
![col\_viz](./figures/col_viz.png)

This verifies that there is no real overlap between the objects as far as color-features are concerned, thereby validating the approach.

## Image Processing Pipeline

Whereas the `ImageProcessor()` class in [the notebook](./code/Rover_Project_Test_Notebook.ipynb) differs slightly from the final version (to be discussed later), the relevant bits of the processing pipeline stayed mostly the same, so I'll simply go through the steps sequentially, as expressed in the notebook.

### Perspective Transform

Initially, the input image undergoes perspective transform from the camera's point of view to a bird's-eye view of the ground-plane projection, like [so](./code/img_proc.py#171):
```python
# in ImageProcessor.__init__()
# self._M = cv2.getPerspectiveTransform(src, dst)
# ...
warped = cv2.warpPerspective(img, self._M, (w,h))# keep same size as input image
```

Essentially, this operation transforms the pixel-coordinates expressed in angular displacements in the camera's perspective and normalizes them so that the newfound coordinates preserve the relationships in *distance*, rather than relatioships in *angle*. This means that now the world coordinates are directly comparable to the pixel locations in the warped image! -- or, at least, given that the angle between the ground plane and the camera's pitch are consistent to the time of calibration.

Here's what the warping process looks like:

Camera Image                                  |  Warped Image (Mine)          | Warped Image (Provided)
:--------------------------------------------:|:-----------------------------:|:-----------------------------:
![a](./calibration_images/example_grid1.jpg)  |  ![b](./figures/warp_bad.png) | ![c](./figures/warp_good.png)

To put this in context, the following are the warp parameters in each of these cases:

Source (Mine)| Source (Provided) | Destination
:-----------:|:-----------------:|:--------:
(119, 97.5)  |(118, 96)          |(155, 144)
(199.5, 97.5)|(200, 96)          |(165, 144)
(302, 141)   |(301, 140)         |(165, 154)
(15, 141)    |(14, 140)          |(155, 154)

Before realizing that these parameters were provided, I manually plotted each of these points; whereas the numbers seem quite similar, the results were quite different (especially for objects that were further away); this was a good example of how even a small difference in the input pixel coordinates can make a large difference in the transformed coordinates, such that a good calibration process was mandatory.

After looking at these results, I reused the provided parameters (right) which seemed to yield a more reasonable projection (namely, the grid-lines were parallel)

### Thresholding

I have already extensively described the thresholding process in the previous section, so I won't labor the point too much. In short, here's how the image looks after thresholding for navigable terrain:

![thresh](./figures/thresh.png)

Note that the grid lines are *outside* of the threshold range, so that they are considered un-navigable. Fortunately, the grid lines are turned off during autonomous navigation, so that no such artifacts are visible! (Even if they did, it could be easily removed through calls like `cv2.dilate()`)

### Coordinate Transforms

After the navigable terrain is identified, the points of interest are identified through `np.where(img)` which returns the locations of non-zero entries as a pair of `(i,j)` coordinates. In order to use these points, we need to convert these coordinates into a frame that best represents where they really are.

#### Rover Coordinates

In the warped image, the rover's location is in the bottom-center, pointing upwards. In order to convert each of these points in the frame of the rover (complying wit hthe standard definition of x-forwards and y-left), we simply apply the following transformation:

```python
def _cvt_rover(self, pi, pj, h, w):
    px, py = float(h)-pi, float(w/2.0)-pj
    return px, py
```

All this means is that `+y=-j` and `+x=-i`, with the center offset of `(x0,y0)=(h,w/2)`. Quite simple!

#### World Coordinates

After this, it is straightforward to convert these coordinates in the `map` frame (or the world-coordinates) by accounting for the position and the yaw-angle of the rover:

```python
def _cvt_world(self, px, py, yaw, tx, ty, ww, wh):
    yaw = np.deg2rad(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    rmat = np.reshape([c,-s,s,c], (2,2))
    wx, wy = np.int32(np.add(np.dot(rmat, [px,py])/self._scale, [[tx],[ty]]))

    wx = np.clip(wx, 0, ww)
    wy = np.clip(wy, 0, wh)
    return wx, wy
```

Note that the positions are scaled by `self._scale`, which is the scaling parameter that represents how many pixels represent a meter. When we were applying the perspective transformation, we were also implicitly accounting for this in the values of the warp destination coordinates, so it needs to be multiplied again. Essentially, this function implements a sequence of rotation, scaling and translation in order to bring the rover's coordinates into the world coordinates.

### Demonstration

Here's the [video](https://youtu.be/TxpBHgyOvME) that demonstrates the full pipeline; it shows the mosaic built by moviepy in the notebook.

# Autonomous Navigation and Mapping

The Demo Video is available [here](https://youtu.be/v4ewCJhmwAo). It's fairly long, so I recommend speeding up the playback rate.

For the most pleasant viewing experience, I recommend downloading the video and playing it with [VLC](https://www.videolan.org/vlc/index.html) at at least 4x playback, as Youtube does not support acceleration greater than 2x.

## Perception

Given the fairly simple visual scene, the perception pipeline did not require significant modifications beyond the basic image-processing and coordinate transformations that had been introduced through the materials in the tutorial.

Most of the improvements on perception was centered around *filtering* the input data, such that the noisy estimation of navigable terrain could at least provide a reasonable accuracy, at no observable cost in the rate of update. At the final stage, each datapoint was filtered by its range from the robot and angular displacement; in addition, updates to the map was limited to when the robot was relatively flat on the ground, where fair perspective transform was guaranteed.

In the below image, the lit region indicates points that are considered valid; the spacing is in 1 meter intervals. Red actually indicates the navigable region (due to BGR ordering in OpenCV), while regions in blue are the obstacles.

![filter](./figures/filter.png)

The filtering code and its visualization can be found [here](./code/img_proc.py).

In the final version, I decided to filter the outputs for when the rover was within +-1 degrees in terms of roll and pitch, as well as the points that were within 7 meters from the rover at less that 75 degrees offset from its yaw. It turned out filtering for angular displacement wasn't mission critical (in fact, the filtered range was greater than the camera's field of view!), but I kept it in there anyways in case it became necessary.

## Decision

### State Machine
The decision portion of the algorithm probably underwent the greatest amount of transformation. In particular, the behavior dictated by the default decision tree was inherently full reactive control, where no high-level planning was involved. This constrained the robot's exploration such that it was unable to survey the whole map.

In order to mitigate this, I decided to implement a [Finite State Machine](https://en.wikipedia.org/wiki/Finite-state_machine) in control of the robot's behavior, which can be found in [rover\_fsm.py](./code/rover_fsm.py). Based on the current map, it will produce and rank a list of frontiers for exploration, use A\* path-finding algorithm to find the route, and travel through local waypoints while trying to avoid obstacles. For extra challenge, I also decided to make the robot pick the samples up, which proved to be no easy task.

Whereas the overall structure of the FSM isn't particularly innovative, I did take the liberty to construct a hierarchical FSM composed of several layers of states. In practice, each state was invoked from several places and overall ended up being a bit of a cobweb, but in principle the logic flows according to the following diagram:

![Diagram](./figures/fsm.svg)

The rover starts by planning its route while going through good candidates for frontiers. In order to prevent the rover from being stuck without a goal, the rover defaults to swerving (reactive obstacle avoidance) while in planning mode. While this may not be ideal in real-world scenarios (because of heavy processing and possible artifacts in the map while moving around, it can be beneficial to "stop and think" while trying to perform high-level planning tasks), it works fairly well under the simulated conditions.

When a good frontier is found, the FSM will issue the path-planner a goal, and find a good global trajectory. After post-processing the trajectory, the waypoints are then sent to the local planner, which then travels along the waypoints one-by-one. If, during its travel, the rock sample is seen, the current exploration will be aborted in favor of collecting the sample; after the sample is picked up, or the destination frontier is reached, the rover will continue the exploration by returning to the planning state.

At any point during navigation, if the rover is stuck, the rover will perform a series of unstuck maneuvers that go through turning counterclockwise, going backwards, turning clockwise, and going forwards. This is not exactly elegant, but works somewhat well in practice.

Note that there is no "terminal" state in the FSM; this is because there is no real need for the rover to stop, under the given constraints. In the current process, the rover will continue to roam the map forever in search of the rock samples. While it is possible to define a terminal state where the number of frontiers is zero (meaning that the exploration is complete), I decided not to risk falling into a situation -- especially in the beginning -- in which the rover temporarily gets stuck in a locally incorrect map with no frontiers, then believing to have completed the exploration. Any other solutions would have required knowledge of the actual map data in the decision-making pipeline, which I wanted to avoid altogether (since it is an intervention from an oracle, with access to ground-truth information).

### Exploration

Reactive control has its limitations in how the robot explores around the environment; in particular, even if there are multiple pathways that had not been previously visited, the rover will behave deterministically and always select the path that it had gone before. This means that without definition of *where* the most desirable target locations are -- i.e. a high-level planner and the definition of a good frontier, the robot will not be able to fully explore the environment.

Rigorously defining the frontier proved to be a challenging task. In a high-level description, a *frontier* for the robot is any region in the map that had not been previously explored; in my case, that meant the *boundary* between identified obstacle and navigable terrain. Among the attempts, I had tried to directly find the contour between the two masks, which proved to be not the most effective strategy. I am not currently aware of a function that will automatically produce the desirable polyline boundary between two color-defined regions without creating artifacts.

I ultimately settled as finding the dilated contour around the navigable region that were *not* covered by the obstacle layers. The frontiers were then sorted by how close it is to the rover, as well as how well it aligns with its current orientation. This was mostly to prevent the rover from oscillating back and forth between distant frontiers, which had been a major time-sink for some iterations.

![Plan](./figures/plan.png)

In the figure, marked in green are the current frontiers; green circle is the current position, red circle is the goal, and the blue path is the global path. Obstacles are marked in magenta.

Overall, this proved to be a good enough approximation of the gaps between walls, that it allowed the rover to complete the map at nearly 98% completion with high fidelity.

## Discussion

### Results

As seen in the video, the rover was successfully able to map and explore **96.5%** of the world, with **83%** fidelity, while collecting **5 rock samples**. The whole mission took about **21 minutes and 20 seconds**.

### Issues

As the rover autonomously navigated and explored the environment, I identified several issues that were not *critical* to the execution, but nonetheless ended up taking up a significant portion of time:

- Local Planner

    In the current implementation, the local planner isn't quite as intelligent as it should be; it simply interpolates between swerving behavior (for obstacle avoidance) and waypoint following behavior (for getting to the final destination). This means that, in principle, there may be edge-cases in which the rover cannot avoid obstacles, or cannot follow the waypoints, when the two behaviors are in severe conflicts. Sometimes it is seen to oscillate in turning, a possible result of such a conflict. It would be nice to have a local planner that can prioritize obstacle avoidance in the presence of immediate danger, and otherwise schedule waypoints intelligently.
    
- Global Planner

    I'm currently using my own implementation of [A\*](code/astar.py) for global path-planning, with a relatively simple heuristic function that penalizes for being too close to obstacles, and a simple euler distance from the goal. The paths are then simplified by [cv2.approxPolyDP()](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#approxpolydp) into an approximate trajectory with fewer waypoints. Whereas this approach is *functional*, ideally the paths produced would be smooth (to prevent stopping to perform in-place turns, which consumes time) and obstacle-free (as sometimes the naive simplification tries to cut across obstacles).
    
- Boulders

    Escaping the boulders in the map proved to be quite a challenge; the success rate of the *unstuck* operations were (empirically) about 90%, although taking tens of seconds at times. Part of the problem was that the camera, when too close to the boulder, would not render them -- which, for the planner, meant that it had a relatively clear path ahead of them. If the planner could always avoid obstacles before even encountering them, this issue would not have surfaced; but since the planner had some trouble trying to resolve obstacle avoidance and navigating to the target location, it would seldom run into rocks and spend some time trying the heuristics to get unstuck.
    
- Semi-Navigable Terrain

    At the intersection between flat ground regions and the obstacles, there exist semi-navigable terrain in which the rover *could* momentarily get stuck, to slow down the mission. Currently, in the processing pipeline, they are treated as "neither obstacle nor navigable" but in reality they are quite a challenge to traverse. To treat this entirely as an obstacle, however, would make planning narrow passageways more difficult. Some heuristic has been implemented to account for buffer zones, but the rover still gets stuck in these boundaries quite often.

### Other Improvements

In the current implementation, the local planner of the rover actually examines the *global* map, which can be problematic in real-world scenarios due to possible latency issues, localization errors, etc. While the local planner has range limits of approximately 6m, and a critical flaw that the projection cannot be trusted when the rover is not nearly flat to the ground plane, it would be nice to be able to integrate local maps to the local planner.

In addition, at the moment the planner does not *revisit* idenfied rock samples if the collection procedure failed; while the odds of this event is not high, it would be ideal if the global planner could be repurposed for two different objectives: exploration of the area (going to frontiers) and collection of rock samples (going to locations). While this would not be a huge modification to the current implementation, the task was not of high-priority, as the rate of failure in collection was fairly low.

## Appendix

As requested, here are the simulation parameter to reproduce the results:

- Resolution : 1024x768, Windowed
- Graphics Quality : Good
