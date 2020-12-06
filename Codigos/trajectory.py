import setup_path 
import airsim
import os
import sys
import math
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from RandomTrajectory import Trajectory_Generation

plt.style.use('ggplot')


class Position:
    def __init__(self, pos):
        self.x = pos.x_val
        self.y = pos.y_val
        self.z = pos.z_val

# Make the drone fly in a circle.
class OrbitNavigator:
    def __init__(self, altitude = -25, speed = 6, snapshots = None):
    
        self.altitude = -25
        self.speed = speed
        self.snapshots = snapshots
        self.snapshot_delta = None
        self.next_snapshot = None
        self.z = None
        self.snapshot_index = 0
        self.takeoff = False # whether we did a take off

        if self.snapshots is not None and self.snapshots > 0:
            self.snapshot_delta = 360 / self.snapshots

        

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.home = self.client.getMultirotorState().kinematics_estimated.position
        print (self.home)
        # check that our home position is stable
        start = time.time()
        count = 0
        while count < 100:
            pos = self.home
            if abs(pos.z_val - self.home.z_val) > 1:                                
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone position is drifting, we are waiting for it to settle down...")
                    start = time
            else:
                count += 1
        final=airsim.Vector3r(0,4,5)
        self.final=final


    def start(self):
        print("arming the drone...")
        self.client.armDisarm(True)
        
        # AirSim uses NED coordinates so negative axis is up.
        start = self.client.getMultirotorState().kinematics_estimated.position
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed: 
            self.takeoff = True
            print("taking off...")
            self.client.takeoffAsync().join()
            start = self.client.getMultirotorState().kinematics_estimated.position
            z = self.altitude
        else:
            print("already flying so we will orbit at current altitude {}".format(start.z_val))
            z = start.self.altitude # use current altitude then

        print("climbing to position: {},{},{}".format(start.x_val, start.y_val, z))
        self.client.moveToPositionAsync(0, 0, z, self.speed).join()
        print ('Drone en posición de inicio')
        position = self.client.getMultirotorState().kinematics_estimated.position
        print (position)
        realdata=[]
        teodata=[]
        n=0
        

        while n<1600:

            v=-(0.15)*position.x_val+4
            alpha=math.atan(-(0.15)*position.x_val+4)

            vy=v*math.sin(alpha)
            vx=v*math.cos(alpha)
            kinematics_estimated = self.client.getMultirotorState().kinematics_estimated
            position=kinematics_estimated.position
            velocity=kinematics_estimated.linear_velocity
            vx_=velocity.x_val
            vy_=velocity.y_val
            vxp=vx+(vx-vx_)*1
            vyp=vy+(vy-vy_)*1
            self.client.moveByVelocityZAsync(vxp, vyp, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom).join()
            print (vx,vy)
            print (vx_,vy_)
            print (vxp,vyp)
            y=-(0.15)*position.x_val*position.x_val+4*position.x_val

            plt.scatter(position.x_val,position.y_val)
            plt.scatter(position.x_val,y)
            plt.pause(0.00001)
            n+=1

            print('***********************')
            
        print("ramping up to speed...")
        count = 0
        self.start_angle = None
        self.next_snapshot = None
        
        # ramp up time
        ramptime = self.radius / 10
        self.start_time = time.time()        


        #self.client.moveByVelocityZAsync(vx, vy, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading))


        self.client.moveToPositionAsync(start.x_val, start.y_val, z, 2).join()

        if self.takeoff:            
            # if we did the takeoff then also do the landing.
            if z < self.home.z_val:
                print("descending")
                self.client.moveToPositionAsync(start.x_val, start.y_val, self.home.z_val - 5, 2).join()

            print("landing...")
            self.client.landAsync().join()

            print("disarming.")
            self.client.armDisarm(False)


    def track_orbits(self, angle):
        # tracking # of completed orbits is surprisingly tricky to get right in order to handle random wobbles
        # about the starting point.  So we watch for complete 1/2 orbits to avoid that problem.
        if angle < 0:
            angle += 360

        if self.start_angle is None:
            self.start_angle = angle
            if self.snapshot_delta:
                self.next_snapshot = angle + self.snapshot_delta
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None            
            self.quarter = False
            return False

        # now we just have to watch for a smooth crossing from negative diff to positive diff
        if self.previous_angle is None:
            self.previous_angle = angle
            return False            

        # ignore the click over from 360 back to 0
        if self.previous_angle > 350 and angle < 10:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False

        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:            
            print("Taking snapshot at angle {}".format(angle))
            self.take_snapshot()
            self.next_snapshot += self.snapshot_delta

        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True

        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            # watch direction this diff is moving if it switches from shrinking to growing
            # then we passed the starting point.
            direction = self.sign(self.previous_diff - diff)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    if self.snapshots <= self.snapshot_index + 1:
                        crossing = True
            self.previous_sign = direction
        self.previous_diff = diff

        return crossing

    def sign(self, s):
        if s < 0: 
            return -1
        return 1

if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("Orbit.py makes drone fly in a circle with camera pointed at the given center vector")
    arg_parser.add_argument("--radius", type=float, help="radius of the orbit", default=10)
    arg_parser.add_argument("--altitude", type=float, help="altitude of orbit (in positive meters)", default=-5)
    arg_parser.add_argument("--speed", type=float, help="speed of orbit (in meters/second)", default=10)
    arg_parser.add_argument("--center", help="x,y direction vector pointing to center of orbit from current starting position (default 1,0)", default="1,0")
    arg_parser.add_argument("--iterations", type=float, help="number of 360 degree orbits (default 3)", default=3)
    arg_parser.add_argument("--snapshots", type=float, help="number of FPV snapshots to take during orbit (default 0)", default=0)    
    args = arg_parser.parse_args(args)    
    nav = OrbitNavigator(args.altitude, args.speed)
    nav.start()
