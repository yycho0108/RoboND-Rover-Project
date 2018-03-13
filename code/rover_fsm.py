import numpy as np
import cv2
from astar import AStar2DGrid as AStar, GDist
from utils import score_frontier, normalize_angle

class RoverFSM(object):
    """ Simple Rover State Machine """
    def __init__(self, rover, state='plan', **sargs):
        self._rover = rover
        self._state = state
        self._sargs = sargs
        self._info = {
                'nomove_cnt' : 0,
                'unstuck_cnt' : 0,
                'stuck_cnt' : 0,
                }
        self._smap = {
                'abort' : self.abort,
                'plan' : self.plan,
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
    def abort(self):
        return 'abort', {}

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

    def plan(self):
        # stop and think ...
        #self.stop()
        res = self.swerve()

        # prioritize getting unstuck ...
        if res[0] == 'unstuck':
            return 'unstuck', {'prv':'plan', 'pargs':{}}
            #return res

        #otherwise keep on planning ...

        # unpack data
        rover = self._rover
        tx, ty = rover.pos
        yaw = rover.yaw
        yaw = normalize_angle(np.deg2rad(yaw))
        map_nav = rover.worldmap[:,:,2]
        map_obs = rover.worldmap[:,:,0]

        # define mapped region ...
        ker = cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3))
        mapped = np.logical_or(
                np.greater(map_nav, 20),
                np.greater(map_obs, 2),
                )
        mapped = 255 * mapped.astype(np.uint8)
        mapped = cv2.erode(mapped, cv2.getStructuringElement(cv2.MORPH_ERODE, (3,3)), iterations=1)
        cnt = cv2.findContours(mapped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        # define frontiers (where nav doesn't meet obs)
        mapped.fill(0)
        goal = None
        if len(cnt) > 0:
            map_nav = cv2.dilate(map_nav, ker, iterations=1)
            cv2.drawContours(mapped, cnt, -1, 255)
            frontier = np.logical_and(map_nav, mapped)
            frontier = 255 * frontier.astype(np.uint8)
            
            fy, fx = frontier.nonzero() #(2,N)

            # basic filter : no obstacles!
            good_goal = (map_obs[fy,fx] <= 0)
            fy = fy[good_goal]
            fx = fx[good_goal]

            if np.size(fy) > 0:
                # order frontiers, good ones at the beginning
                fx, fy = score_frontier(tx, ty, yaw, fx, fy)

            for goal in zip(fx, fy):
                # try goals sequentially
                res = self.moveto(goal)
                if res[0] != 'abort':
                    # good plan, go forth!
                    return res

        # keep planning
        #return 'swerve', {}
        return 'plan', {}

    def moveto(self, target):
        """
        Get the rover to a destination. Aborts if impossible.
        TODO : actually abort if impossible
        """

        #rover = self._rover
        #src = rover.pos
        #trg = target
        #dx, dy = np.subtract(target, rover.pos)
        #dist = np.sqrt(dx**2+dy**2)
        #ang = np.arctan2(dy, dx) - np.deg2rad(rover.yaw)
        #ang = (ang + np.pi) % (2*np.pi) - np.pi

        return self.moveto_global(target)

        #if dist <= 5.0 and np.abs(ang) < np.pi / 2.0:
        #    # close & front-ish
        #    return self.moveto_local(target)
        #else:
        #    return self.moveto_global(target)



    def moveto_global(self, target):
        """
        Compose global plan from current position and target.
        Simply uses A* path-planning for right now.
        The path is sort-of-pushed to navigable area by Sobel(),
        although it doesn't seem to work that well.
        """
        # compute ...
        rover = self._rover

        #map_obs = cv2.dilate(
        #        self._rover.worldmap[:,:,0].astype(np.uint8),
        #        cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3)),
        #        iterations=1
        #        ).astype(np.float32)
        #map_obs /= 5.0

        # alternatively:
        map_obs = np.greater(self._rover.worldmap[:,:,0], 2).astype(np.float32)
        #map_obs = np.less(self._rover.worldmap[:,:,2], 10)

        #map_obs = cv2.erode(map_obs.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ERODE, (3,3)))
        #map_obs = cv2.dilate(map_obs.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3)))
        #map_obs = map_obs.astype(np.bool)
        src = tuple(np.int32(rover.pos))
        dst = tuple(np.int32(target))
        map_obs[src[1], src[0]] = 0.0 # make sure init cell is open?
        h = GDist(dst, map_obs, scale=10.0)
        astar = AStar(map_obs, src, dst, h=h)#, h=GDist(x1=dst[::-1], o=map_obs, scale=1.0))
        _, path = astar() # given as (x,y)
        # simplify path
        if path is None:
            # TODO : 
            # if (realized_cannot_get_to_target) then
            #   ask_for_new_target()
            # No Path! Abort
            # TODO : Fix
            rover.goal = None
            return 'abort', {}
        else:
            path = cv2.approxPolyDP(path, 2.0, closed=False)[:,0,:]

            ## push out from wall
            dy = cv2.Sobel(map_obs, cv2.CV_8UC1, 0, 1, ksize=5) / 5.
            dx = cv2.Sobel(map_obs, cv2.CV_8UC1, 1, 0, ksize=5) / 5.
            delta = np.stack((dx,dy), axis=-1)
            dp = delta[path[:,0], path[:,1]]
            #path = np.int32(path - dp)
            ####################

            # registered as current goal
            rover.goal = target
            rover.path = np.int32(path - dp)
            rover.p0 = path
            # TODO : make sure poly approximation doesn't cross obstacles

        return 'moveto_local', {'path' : path, 'phase' : 'turn'}

    def moveto_local(self, path=None, phase='turn'):
        """
        Follow local waypoints.
        Currently, moveto_local() doesn't handle scenarios
        such as impossible plans, getting stuck, or hitting obstacles.
        """
        rover = self._rover

        # TODO : 
        # if (realized_cannot_get_to_point) then
        #   ask_for_new_global_plan()

        # logic for re-routing local path
        # to avoid obstacles - like rocks, for instance.

        if len(path) == 0:
            # completed goal! start planning.
            rover.goal = None
            return 'plan', {}
            #return 'swerve', {}

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
            dadr = np.abs(da) / dr #15
            if np.abs(da) <= np.deg2rad(10):#~+-10 deg.
                # accept current angle + forwards
                return 'moveto_local', {'path' : path, 'phase' : 'forward'}
            steer = np.clip(np.rad2deg(da), -15.0, 15.0)
            self.turn(steer)
        elif phase  == 'forward':
            # turn less responsively
            steer = np.clip(np.rad2deg(da), -15.0, 15.0)
            self.move(steer=steer)
            # TODO : check stuck-ness here

        if self._info['nomove_cnt'] > 120:
        #if self.check_stuck():
            self._info['unstuck_cnt'] += 1
            if self._info['unstuck_cnt'] > 1: # try 1 time
                self._info['unstuck_cnt'] = 0
                # really stuck. ask for a new plan!
                return 'moveto_local', {'path' : [], 'phase' : 'turn'}
            else:
                # try to get self unstuck.
                return 'unstuck', {'prv' : 'moveto_local', 'pargs' : {'path' : path, 'phase' : 'forward'}}

        return 'moveto_local', {'path' : path, 'phase' : phase}

    def swerve(self):
        """ default implementation; follow navigable angles """
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

    def show(self):
        rover = self._rover
        tx, ty = rover.pos
        yaw = rover.yaw
        yaw = normalize_angle(np.deg2rad(yaw))
        map_nav = rover.worldmap[:,:,2]

        map_obs = rover.worldmap[:,:,0]

        cimg = (np.logical_and(
            np.greater(map_nav, 20),
            np.greater(map_obs, 20)) * 255).astype(np.uint8)
        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
        
        if rover.goal is not None:
            cv2.circle(cimg, tuple(np.int_(rover.pos)), 2, [0.0, 255, 0.0])
            cv2.circle(cimg, tuple(np.int_(rover.goal)), 2, [0.0,0.0,255])
        if rover.path is not None:
            for (p0, p1) in zip(rover.path[:-1], rover.path[1:]):
                x0,y0 = p0
                x1,y1 = p1
                cv2.line(cimg, (x0,y0), (x1,y1), (255,0,0), 1)
        #if rover.p0 is not None:
        #    for (p0, p1) in zip(rover.p0[:-1], rover.p0[1:]):
        #        x0,y0 = p0
        #        x1,y1 = p1
        #        #cv2.line( (y0,x0), (y1,x1), (128)
        #        cv2.line(cimg, (x0,y0), (x1,y1), (255,255,0), 1)

        # cimg will show mission status; position, goal, boundary, path.
        cv2.imshow('cimg', np.flipud(cimg))
        cv2.waitKey(10)

    def run(self):
        """ Run the State Machine """
        print 'fsm state : ', self._state
        sfn = self._smap[self._state]
        res = sfn(**self._sargs)
        self._state, self._sargs = res

        viz = True
        if viz:
            self.show()


        ## responding to goals; comment out below to run without high-level behavior
        ## (( goals are created in img_proc.py ))
        #if not (self._state is 'unstuck'):
        #    if self._rover.next_goal is not None: # has a goal
        #        if not (self._state.startswith('moveto')): # avoid interrupting
        #            self._state = 'moveto'
        #            self._sargs = {'target' : self._rover.next_goal}
