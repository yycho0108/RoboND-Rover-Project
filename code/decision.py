import numpy as np

class Planner(object):
    def __init__(self):
        pass
    @staticmethod
    def local_plan(Rover,
            init, init_yaw, goal,
            nav_r, nav_a, global_map=None,
            ):
        # considers polar local map, returns target vel + target angle
        # considers ackermann drive kinematics?
        #alpha = tan(steer)/L * vel

        dx, dy = np.subtract(goal, init)
        target_yaw = np.rad2deg(np.arctan2(dy,dx))
        target_dist = np.linalg.norm([dx,dy])

        da = (target_yaw - init_yaw + 540) % 360 - 180
        da_r = np.clip(np.deg2rad(da), -np.deg2rad(60), np.deg2rad(60))
        atol = np.deg2rad(10)

        aand = np.logical_and

        # apply range filter ...
        valid_idx = aand(nav_r < 20.0, np.abs(nav_a) < np.deg2rad(75))

        nav_r = nav_r[valid_idx]
        nav_a = nav_a[valid_idx]

        # apply
        front_idx = aand(nav_a > -2*atol, nav_a < 2*atol)
        nav_r_front = nav_r[front_idx]
        nav_a_front = nav_a[front_idx]

        good_idx = aand(nav_a > da_r-atol, nav_a < da_r+atol)
        nav_r_good = nav_r[good_idx]
        nav_a_good = nav_a[good_idx]

        print 'Good : {}'.format(len(nav_r_good))

        path_blocked = len(nav_r_front) < 20

        if path_blocked:
            print 'block!'
            #Rover.mode = 'swerve'
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = 15# * np.sign(da) #np.clip(da, -15, 15)# * np.sign(da)
            #try_to_swerve()
            return False
        else:
            # go right ahead!
            if np.abs(da) > 10.0:
                # turn first
                Rover.throttle = 0.0
                if Rover.vel > 0.2:
                    Rover.brake = Rover.brake_set
                else:
                    Rover.brake = 0
                Rover.steer = np.clip(da, -15, 15)
            else:
                # go towards goal
                # with small angle adjustments
                Rover.brake = 0.0
                target_vel = np.clip(target_dist * 0.3, 0, Rover.max_vel)
                if Rover.vel < target_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else:
                    Rover.throttle = 0
                Rover.steer = np.clip(da * 0.5, -5, 5)
            return True



# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with

    if Rover.mode == 'goal':
        if len(Rover.path) <= 0:
            Rover.mode = 'stop'
            return Rover
        target = Rover.path[0]
        (dx, dy) = np.subtract(target, Rover.pos)
        target_yaw = np.rad2deg(np.arctan2(dy, dx))
        #print 'tp | p', target, Rover.pos
        #print 'ty | y', target_yaw, Rover.yaw
        target_dist = np.linalg.norm([dx,dy])
        if target_dist < 3.0: #+-3.0m
            Rover.path = Rover.path[1:]
            return Rover

        suc = Planner.local_plan(
                Rover, Rover.pos, Rover.yaw,
                target, Rover.nav_dists, Rover.nav_angles)

        if suc:
            return Rover
        else:
            Rover.mode='stop'
            return Rover


        #delta_yaw = (target_yaw - Rover.yaw)
        #delta_yaw = (delta_yaw+540)%360 - 180 # +-180
        #print delta_yaw
        #if np.abs(delta_yaw) > 10.0:
        #    # turn first
        #    Rover.throttle = 0.0
        #    if Rover.vel > 0.2:
        #        Rover.brake = Rover.brake_set
        #    else:
        #        Rover.brake = 0
        #    Rover.steer = np.clip(delta_yaw, -15, 15)
        #else:
        #    Rover.brake = 0.0
        #    # turn done
        #    if Rover.vel < Rover.max_vel:
        #        # Set throttle value to throttle setting
        #        Rover.throttle = Rover.throttle_set
        # follow path ...
        return Rover

    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

