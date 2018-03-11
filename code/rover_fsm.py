import numpy as np
import cv2
from astar import AStar2DGrid as AStar

class RoverFSM(object):
    """ Simple Rover State Machine """
    def __init__(self, rover, state='swerve', **sargs):
        self._rover = rover
        self._state = state
        self._sargs = sargs
        self._info = {
                'nomove_cnt' : 0,
                'stuck_cnt' : 0,
                }
        self._smap = {
                'moveto' : self.moveto,
                'moveto_local' : self.moveto_local,
                'moveto_global' : self.moveto_global,
                'unstuck' : self.unstuck,
                'swerve' : self.swerve,
                'pickup' : self.pickup
                }

    """ status utility functions """
    def check_move(self):
        """
        Check whether or not the rover is moving;
        returns True if rover is moving.
        """
        rover = self._rover
        return (np.abs(rover.vel) > 0.1)

    def check_stuck(self):
        """
        Check whether or not the rover is stuck,
        i.e. in unnavigable terrain;
        returns True is rover is stuck.
        """
        rover = self._rover
        return len(rover.nav_angles) < rover.go_forward or self._info['nomove_cnt'] > 120


    def check_obs(self, ang, atol=np.deg2rad(10)):
        """
        returns True if obstacle exists in a direction.
        ang : angle to search obstacle, in radians.
        atol : search width tolerance, in radians (default=10 deg.)
        """
        rover = self._rover
        nav_a = rover.nav_angles

        ang = (ang + np.pi) % (2*np.pi) - np.pi
        ang = np.clip(ang, -np.deg2rad(60), np.deg2rad(60))
        good_idx = np.logical_and(nav_a > ang-atol, nav_a < ang + atol)
        return len(good_idx) < rover.go_forward

    """ high-level actions """
    def moveto(self, target):
        """ Get the rover to a destination. Aborts if impossible. """
        rover = self._rover
        src = rover.pos
        trg = target
        dx, dy = np.subtract(target, rover.pos)
        dist = np.sqrt(dx**2+dy**2)
        ang = np.arctan2(dy, dx) - np.deg2rad(rover.yaw)
        ang = (ang + np.pi) % (2*np.pi) - np.pi

        return self.moveto_global(target)

        #if dist <= 5.0 and np.abs(ang) < np.pi / 2.0:
        #    # close & front-ish
        #    return self.moveto_local(target)
        #else:
        #    return self.moveto_global(target)

    def unstuck(self, prv, pargs):
        """
        Get the rover out of a stuck state.
        Returns to 'prv' state after it regains mobility.
        """
        stuck = self.check_stuck()

        period = 400
        if stuck:
            self._info['stuck_cnt'] += 1
            if self._info['stuck_cnt'] % period < period * 0.2:
                # try turning, 20% of the time
                self.turn(steer=15)
            elif self._info['stuck_cnt'] % period < period * 0.5:
                # try moving back, 30% of the time
                self.move(target_vel = -self._rover.max_vel)
            elif self._info['stuck_cnt'] % period < period * 0.7:
                # try turning, 20% of the time
                self.turn(steer=-15)
            else:
                # try moving forwards, 30% of the time
                self.move(target_vel = +self._rover.max_vel)

            return 'unstuck', {'prv': prv, 'pargs' : pargs}
        else:
            # clear flag
            self._info['stuck_cnt'] = 0
            # go back to whatever previous step was.
            return prv, pargs 

    def moveto_global(self, target):
        # compute ...
        rover = self._rover

        map_obs = np.greater(self._rover.worldmap[:,:,0], 2)
        #map_obs = cv2.dilate(map_obs.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3)))
        #map_obs = map_obs.astype(np.bool)
        src = tuple(np.int32(rover.pos))
        dst = tuple(np.int32(target))
        astar = AStar(map_obs, src, dst)
        _, path = astar()
        # simplify path
        if path is None:
            # TODO : 
            # if (realized_cannot_get_to_target) then
            #   ask_for_new_target()
            # No Path! Abort
            # TODO : Fix
            rover.goal = None
            return 'swerve', {}
        else:
            path = cv2.approxPolyDP(path, 3.0, closed=False)[:,0,:]
            # TODO : makesure poly approximation doesn't cross obstacles

        return 'moveto_local', {'path' : path, 'phase' : 'turn'}

    def moveto_local(self, path=None, phase='turn'):
        rover = self._rover

        # when I write a proper local planner:
        #if self._info['moveto_local_path'] is None:
        #    astar = AStar(mo, (tx,ty), goal)
        #    _, path = astar()
        #    self._info['moveto_local_path'] = path

        # TODO : 
        # if (realized_cannot_get_to_point) then
        #   ask_for_new_global_plan()

        if len(path) == 0:
            # TODO : fix this when done
            # completed goal!
            rover.goal = None
            return 'swerve', {}

        target = path[0]
        tx, ty = target
        dx, dy = np.subtract(target, rover.pos)

        da = np.arctan2(dy, dx) - np.deg2rad(rover.yaw)
        da = (da + np.pi) % (2*np.pi) - np.pi
        dr = np.sqrt(dx**2+dy**2)

        if dr <= 2.0:
            # accept current position + move on
            return 'moveto_local', {'path' : path[1:], 'phase' : 'turn'}

        if phase == 'turn':
            if np.abs(da) <= np.deg2rad(10):
                # accept current angle + forwards
                return 'moveto_local', {'path' : path, 'phase' : 'forward'}
            steer = np.clip(np.rad2deg(da), -15.0, 15.0)
            self.turn(steer)
        elif phase  == 'forward':
            # turn less responsively
            steer = np.clip(np.rad2deg(da / 2.0), -15.0, 15.0)
            self.move(steer=steer)
            return 'unstuck', {'prv' : 'moveto_local', 'pargs' : {'path' : path, 'phase' : 'forward'}}
            # TODO : check stuck-ness here

        if self._info['nomove_cnt'] > 120:
            # stuck-ish. ask for a new plan!
            return 'moveto_local', {'path' : [], 'phase' : 'turn'}

        return 'moveto_local', {'path' : path, 'phase' : phase}

        # target direction
        #if target is not None:
        #    tx, ty = target
        #    dx, dy = np.subtract(target, rover.pos)

        #    dd = np.sqrt(dx**2 + dy**2)
        #    if dd < 2.0:
        #        self._rover.goal = None
        #        return 'moveto_local', {'target' : None}
        #    elif self._rover.worldmap[ty,tx,0] > 5:
        #        # abort
        #        self._rover.goal = None
        #        return 'moveto_local', {'target' : None}

        #    da = np.arctan2(dy, dx) - np.deg2rad(rover.yaw)
        #    da = (da + np.pi) % (2*np.pi) - np.pi

        #    if self.check_obs(da):
        #        print 'swerve-obs'
        #        # prioritize obstacle avoidance
        #        # move forwards max-speed, but with angle in mind.
        #        steer = np.clip(np.mean(rover.nav_angles * 180/np.pi), -15, 15)
        #        self.move(steer=steer)
        #    else:
        #        print 'swerve-clear'
        #        # no obstacle in sight
        #        if np.abs(da) > np.deg2rad(45): # +-60 deg.
        #            print 'swerve-turn'
        #            # turn first if target is to your back
        #            #steer = np.clip(da * 10, -15, 15)
        #            steer = np.clip(np.mean(rover.nav_angles * 180/np.pi), -15, 15)
        #            #self.turn(steer)
        #            self.move(steer=steer)
        #        else:
        #            print 'swerve-fw'
        #            steer = np.clip(np.mean(rover.nav_angles * 180/np.pi), -15, 15)
        #            self.move(steer=steer)
        #            #steer = np.clip(da * 20, -15, 15)
        #            #self.move(steer=steer)
        #    return 'moveto_local', {'target' : target}

    def swerve(self):
        rover = self._rover

        # Check the extent of navigable terrain
        if self.check_stuck():
            return 'unstuck', {'prv':'swerve', 'pargs' : {}}
        
        else:
            # default behavior; follow the terrain-ish.
            steer = np.clip(np.mean(rover.nav_angles * 180/np.pi), -15, 15)
            self.move(steer=steer)
            return 'swerve', {}

    """ primitive actions """
    def stop(self):
        rover = self._rover
        rover.throttle = 0
        rover.brake = rover.brake_set
        rover.steer = 0

    def unstop(self):
        rover = self._rover
        rover.brake = 0

    def turn(self, steer=-15):
        """ Performs pure in-place turn. """
        # TODO : option to set target angle
        if self.check_move():
            self.stop()
        else:
            self.unstop()
            rover = self._rover
            rover.throttle=0
            rover.steer=steer# Could be more clever here about which way to turn

    def pickup(self):
        if Rover.near_sample:
            if Rover.vel == 0 and not Rover.picking_up:
                Rover.send_pickup = True
            return 'pickup', {}
        else:
            return 'swerve', {}

    def move(self, target_vel=None, steer=0):
        """ Go mainly forwards """
        self.unstop()
        
        if not self.check_move():
            self._info['nomove_cnt'] += 1
        else:
            self._info['nomove_cnt'] = 0

        rover = self._rover
        if target_vel is None: # fill in target
            target_vel = rover.max_vel

        if target_vel < rover.vel and target_vel < 0:
            rover.throttle = -rover.throttle_set
        elif target_vel > rover.vel and target_vel > 0:
            rover.throttle = rover.throttle_set
            #np.abs(rover.vel) < np.abs(target_vel):
            # Set throttle value to throttle setting
            # TODO : PID here? probably unnecessary
            #rover.throttle = sgn * rover.throttle_set
        else: # Else coast
            rover.throttle = 0
        rover.steer = steer

    def run(self):
        print 'fsm state : ', self._state
        sfn = self._smap[self._state]
        res = sfn(**self._sargs)
        self._state, self._sargs = res

        # global state tracking ...
        
        # responding to planner ...
        if not (self._state is 'unstuck'):
            if self._rover.goal is not None: # has a goal
                if not (self._state.startswith('moveto')): # avoid interrupting
                    self._state = 'moveto'
                    self._sargs = {'target' : self._rover.goal}
