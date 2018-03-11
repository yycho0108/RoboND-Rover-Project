import numpy as np
from rover_fsm import RoverFSM

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function

fsm = None
def decision_step(rover):
    global fsm
    if fsm is None:
        fsm = RoverFSM(rover)
    fsm.run()
    return rover 
