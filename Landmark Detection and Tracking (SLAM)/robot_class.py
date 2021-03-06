# 
#  Udacity Computer Vision Nanodegree - Uljan Sinani, 02/09/2020
#

# import libraries
from math import *
import random


### ------------------------------------- ###
# Below, is the robot class
#
# This robot lives in 2D, x-y space, and its motion is
# pointed in a random direction, initially.
# It moves in a straight line until it comes close to a wall 
# at which point it stops.
#
# For measurements, it  senses the x- and y-distance
# to landmarks. This is different from range and bearing as
# commonly studied in the literature, but this makes it much
# easier to implement the essentials of SLAM without
# cluttered math.
#
class robot:
    # init:
    #   creates a robot with the specified parameters and initializes
    #   the location (self.x, self.y) to the center of the world
    def __init__(self, world_size = 100.0, measurement_range = 30.0,
                 motion_noise = 1.0, measurement_noise = 1.0):
        self.measurement_noise = 0.0
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.x = world_size / 2.0
        self.y = world_size / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0
    
    
    # returns a positive, random float
    def rand(self):
        return random.random() * 2.0 - 1.0
        
    # move: attempts to move robot by dx, dy. If outside world
    #       boundary, then the move does nothing and instead returns failure
    #
    def move(self, dx, dy):
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise
        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        else:
            self.x = x
            self.y = y
            return True

    ## TODO: paste your complete the sense function, here
    ## make sure the indentation of the code is correct
    def sense(self):
        measurements = []
        
        ## TODO: iterate through all of the landmarks in a world
        ## TODO: For each landmark
        ## 1. compute dx and dy, the distances between the robot and the landmark
        ## 2. account for measurement noise by *adding* a noise component to dx and dy
        ##    - The noise component should be a random value between [-1.0, 1.0)*measurement_noise
        ##    - Feel free to use the function self.rand() to help calculate this noise component
        ## 3. If either of the distances, dx or dy, fall outside of the internal var, measurement_range
        ##    then we cannot record them; if they do fall in the range, then add them to the measurements list
        ##    as list.append([index, dx, dy]), this format is important for data creation done later
        
        ## TODO: return the final, complete list of measurements
        dx = 0
        dy = 0

        # iterate through all of the landmarks in a world
        for index in range(self.num_landmarks):
            landmark_dx = self.landmarks[index][0]
            landmark_dy = self.landmarks[index][1]
            
        # 1. compute dx and dy, the distances between the robot and the landmark
            dx = landmark_dx - self.x
            dy = landmark_dy - self.y
        
        # 2. account for measurement noise by *adding* a noise component to dx and dy
            noise = self.rand() * self.motion_noise
            dx = dx + noise
            dy = dy + noise
            
        # 3. add to the measurements list if either of the distances dx or dy falls inside of measurement_range
        # I have to keep in mind that dx and dy can be negative thus, I have to ensure the values fall within the required measurement range. 
        # Using the abs(dx) I can mitigate this problem by taking only the absolute values of dx and dy.
        # Take the absolute values and use only for comparison, then add the actual values to the list.
            abs_dx, abs_dy = abs(dx), abs(dy)
            if (abs_dx <= self.measurement_range) and (abs_dy <= self.measurement_range):
                measurements.append([index, dx, dy])
            
        # return the final complete list of measurements
        return measurements

    # --------
    # make_landmarks:
    # make random landmarks located in the world
    #
    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for i in range(num_landmarks):
            self.landmarks.append([round(random.random() * self.world_size),
                                   round(random.random() * self.world_size)])
        self.num_landmarks = num_landmarks

    # called when print(robot) is called; prints the robot's location
    def __repr__(self):
        return 'Robot: [x=%.5f y=%.5f]'  % (self.x, self.y)


####### END robot class #######
