# Project : Search and Sample Return

## Write-Up

## Notebook Analysis

### Color Selection

Unlike the examples provided, I first converted the input image in the HSV colorspace in order to minimize sensitivity to brightness, or saturation. The three types of environmental markers (ground plane, obstacles and rocks) were clearly distinguishable in the RGB colorspace as well, so it was mostly a matter of personal preference.

To this end, I created a [separate script](./code/helpers/find_color.py) that took in an image and basically allowed the user to find lower and upper bounds by clicking on multiple sample points.

The perspective transform was simply adopted from the [notebook](./code/Rover_Project_Test_Notebook.ipynb); after some experimentation, it appeared that the numbers provided by the lecture seemed to yield a more accurate projection, so I reused the values in all subsequent experiments.

## Autonomous Navigation and Mapping

The Demo Video is available [here](https://youtu.be/v4ewCJhmwAo). It's fairly long, so I recommend speeding up the playback rate.

### Perception

Given the fairly simple visual scene, the perception pipeline did not require significant modifications beyond the basic image-processing and coordinate transformations that had been introduced through the materials in the tutorial.

Most of the improvements on perception was centered around *filtering* the input data, such that the noisy estimation of navigable terrain could at least provide a reasonable accuracy, at no observable cost in the rate of update. At the final stage, each datapoint was filtered by its range from the robot and angular displacement; in addition, updates to the map was limited to when the robot was relatively flat on the ground (+- 1.0 degrees in pitch and roll), where fair perspective transform was guaranteed.

# Polar Grid figure, etc.

### Decision

The decision portion of the algorithm probably underwent the greatest amount of transformation. In particular, the behavior dictated by the default decision tree was inherently full reactive control, where no high-level planning was involved. This constrained the robot's exploration such that it was unable to survey the whole map.

In order to mitigate this, I decided to implement a [Finite State Machine](https://en.wikipedia.org/wiki/Finite-state_machine) in control of the robot's behavior, which can be found in [rover\_fsm.py](./code/rover_fsm.py). Based on the current map, it will produce and rank a list of frontiers for exploration, use A\* path-finding algorithm to find the route, and travel through local waypoints while trying to avoid obstacles. For extra challenge, I also decided to make the robot pick the samples up, which proved to be no easy task.

Whereas the overall structure of the FSM isn't particularly innovative, I did take the liberty to construct a hierarchical FSM composed of several layers of states. In practice, each state was invoked from several places and overall ended up being a bit of a cobweb, but in principle the logic flows according to the following diagram:

![Diagram]()

## Appendix

As requested, here are the simulation parameter to reproduce the results:

- Resolution : 1024x768, Windowed
- Graphics Quality : Good




