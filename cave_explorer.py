#!/usr/bin/env python3

import rospy
import roslib
import math
import cv2 # OpenCV2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from enum import Enum


def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle


def pose2d_to_pose(pose_2d):
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORER = 6
    # Add more!

class CaveExplorer:
    def __init__(self):

        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False
        self.grid_map_ = None

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.reached_closest_frontier_ = False
        self.finised_exploring = False
        self.rotate_ = False
        self.goal_counter_ = 0 # gives each goal sent to move_base a unique ID

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")        

        # Advertise "cmd_vel" publisher to control the robot manually -- though usually we will be controller via the following action client
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Publisher for the camera detections
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # Read in computer vision model (simple starting point)
        self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

        # Subscribe to the map topic
        self.grid_map_sub_ = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        # Wait until the map is received
        rospy.loginfo("Waiting for map...")

        while not rospy.is_shutdown():
            if self.grid_map_ is not None:
                rospy.loginfo("Map received!")
                break
            rospy.sleep(0.1)


    def get_pose_2d(self):

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]

        qw = rot[3];
        qz = rot[2];

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw));

        rospy.loginfo(f"Current Robot Pose: x={pose.x}, y={pose.y}, theta={pose.theta}")

        return pose


    def image_callback(self, image_msg):
        # This method is called when a new RGB image is received
        # Use this method to detect artifacts of interest
        #
        # A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
        # adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/

        # Copy the image message to a cv image
        # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Create a grayscale version, since the simple model below uses this
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Retrieve the pre-trained model
        stop_sign_model = self.computer_vision_model_

        # Detect artifacts in the image
        # The minSize is used to avoid very small detections that are probably noise
        detections = stop_sign_model.detectMultiScale(image, minSize=(20,20))

        # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
        # You may want to communicate more information
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "artifact_found_" doesn't need a mutex because it's an atomic
        num_detections = len(detections)

        if num_detections > 0:
            self.artifact_found_ = True
        else:
            self.artifact_found_ = False

        # Draw a bounding box rectangle on the image for each detection
        for(x, y, width, height) in detections:
            cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        #rospy.loginfo('image_callback')
        #rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))

    def map_callback(self, map_data):
        # This method is called when a new map is received to update the map
        self.grid_map_ = map_data
        #rospy.loginfo('New map received!')


    def planner_move_forwards(self, action_state):
        # Simply move forward by 10m

        # Only send this once before another action
        if action_state == actionlib.GoalStatus.LOST:

            pose_2d = self.get_pose_2d()

            rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Move forward 10m
            pose_2d.x += 10 * math.cos(pose_2d.theta)
            pose_2d.y += 10 * math.sin(pose_2d.theta)

            rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal to move forward...')
            self.move_base_action_client_.send_goal(action_goal.goal)


    def planner_go_to_first_artifact(self, action_state):
        # Go to a pre-specified artifact (alien) location

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 18.0
            pose_2d.y = 25.0
            pose_2d.theta = -math.pi/2

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal to artifact...')
            self.move_base_action_client_.send_goal(action_goal.goal)


    def planner_return_home(self, action_state):
        # Go to the origin

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 0
            pose_2d.y = 0
            pose_2d.theta = 0

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal of returning home...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def planner_random_walk(self, action_state):
        # Go to a random location, which may be invalid

        min_x = -5
        max_x = 50
        min_y = -5
        max_y = 50

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            pose_2d = Pose2D()
            pose_2d.x = random.uniform(min_x, max_x)
            pose_2d.y = random.uniform(min_y, max_y)
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal to random walk...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def planner_random_goal(self, action_state):
        # Go to a random location out of a predefined set

        # Hand picked set of goal locations
        random_goals = [[53.3,40.7],[44.4, 13.3],[2.3, 33.4],[9.9, 37.3],[3.4, 18.5],[6.0, 0.4],[28.3, 11.8],[43.7, 12.8],[38.9,43.0],[47.4,4.7],[31.5,3.2],[36.6,32.5]]

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            idx = random.randint(0,len(random_goals)-1)
            pose_2d = Pose2D()
            pose_2d.x = random_goals[idx][0]
            pose_2d.y = random_goals[idx][1]
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending random goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)

       
    def planner_to_frontiers(self, action_state):
        # Only proceed if the robot isn't already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:
            rospy.loginfo("Exploring the cave...")

            width = self.grid_map_.info.width
            height = self.grid_map_.info.height
            resolution = self.grid_map_.info.resolution
            origin_x = self.grid_map_.info.origin.position.x
            origin_y = self.grid_map_.info.origin.position.y
            map_data = np.array(self.grid_map_.data).reshape((height, width))
            frontiers = []
            

            # Detect frontiers by checking neighboring unknown space
            for y in range(height):
                for x in range(width):
                    if map_data[y, x] == 0:  # 0 means free space
                        neighbours = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0) and 0 <= x + dx < width and 0 <= y + dy < height]
                        if any(map_data[ny, nx] == -1 for nx, ny in neighbours):
                            wx, wy = x * resolution + origin_x, y * resolution + origin_y
                            frontiers.append((wx, wy))
            rospy.loginfo(f"Detected {len(frontiers)} frontiers and largest is {max(frontiers, key=len)}")
            
            # Group nearby frontiers
            grouped_frontiers = []
            queue = [(int((wx - origin_x) / resolution), int((wy - origin_y) / resolution)) for wx, wy in frontiers]
            visited = []  # Initialize visited here

            while queue:
                current = queue.pop(0)
                
                if current in visited:
                    continue
                
                # Start a new group
                current_group = []
                to_visit = [current]

                while to_visit:
                    frontier = to_visit.pop(0)
                    if frontier in visited:
                        continue
                    
                    visited.append(frontier)
                    current_group.append(frontier)

                    # Get neighbors
                    neighbours = [(frontier[0] + dx, frontier[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                if (dx != 0 or dy != 0) and 
                                0 <= frontier[0] + dx < width and 
                                0 <= frontier[1] + dy < height]
                    
                    for neighbour in neighbours:
                        if neighbour in queue:
                            queue.remove(neighbour)
                            to_visit.append(neighbour)

                # Check if the group size is greater than 10
                if len(current_group) > 500:
                    grouped_frontiers.append(current_group)

            
            if frontiers:
                robot_pose = Pose2D()
                robot_x = robot_pose.x
                robot_y = robot_pose.y

                closest_frontier_group = None
                min_distance = float('inf')

                # Find the closest frontier group
                for group in grouped_frontiers:
                    centroid_x = sum([cell[0] for cell in group]) / len(group)
                    centroid_y = sum([cell[1] for cell in group]) / len(group)

                    # Convert centroid to world coordinates
                    wx, wy = centroid_x * resolution + origin_x, centroid_y * resolution + origin_y

                    # Compute the distance between robot and frontier group
                    distance = ((wx - robot_x) ** 2 + (wy - robot_y) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        closest_frontier_group = group

                centroid_x = sum([cell[0] for cell in closest_frontier_group]) / len(closest_frontier_group)
                centroid_y = sum([cell[1] for cell in closest_frontier_group]) / len(closest_frontier_group)
                wx, wy = centroid_x * resolution + origin_x, centroid_y * resolution + origin_y

                # Send a goal to "move_base"
                pose_2d = Pose2D()
                pose_2d.x = wx
                pose_2d.y = wy

                action_goal = MoveBaseActionGoal()
                action_goal.goal.target_pose.header.frame_id = "map"
                action_goal.goal_id = self.goal_counter_
                self.goal_counter_ += 1
                action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

                rospy.loginfo(f'Sending goal to the closest frontier at ({wx}, {wy}) with size {len(closest_frontier_group)}')
                self.move_base_action_client_.send_goal(action_goal.goal)

            # When no frontiers is left
            else:
                self.finished_exploring = True
                rospy.loginfo("No frontiers detected. Exploration complete.")


    def main_loop(self):

        while not rospy.is_shutdown():

            #######################################################
            # Get the current status
            # See the possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            #rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            #rospy.loginfo('action_state number:' + str(action_state))

            if (self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully reached first artifact!")
                self.reached_first_artifact_ = True
            if self.planner_type_ == PlannerType.FRONTIER_EXPLORER and action_state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Explored a frontier.")
            if (self.planner_type_ == PlannerType.RETURN_HOME) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully returned home!")
                self.returned_home_ = True




            #######################################################
            # Select the next planner to execute
            # Update this logic as you see fit!
            # self.planner_type_ = PlannerType.MOVE_FORWARDS
            if not self.finised_exploring:
                self.planner_type_ = PlannerType.FRONTIER_EXPLORER
            else:
                if not self.returned_home_:  # If we haven't returned home yet
                    self.planner_type_ = PlannerType.RETURN_HOME
                    rospy.loginfo("No more frontiers. Returning home...")
                else:
                    rospy.loginfo("All exploration complete and home returned. Exiting.")
                    break


            #######################################################
            # Execute the planner by calling the relevant method
            # The methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            # rospy.loginf("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.MOVE_FORWARDS:
                self.planner_move_forwards(action_state)
            elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
                self.planner_go_to_first_artifact(action_state)
            elif self.planner_type_ == PlannerType.FRONTIER_EXPLORER:
                self.planner_to_frontiers(action_state)
            elif self.planner_type_ == PlannerType.RETURN_HOME:
                self.planner_return_home(action_state)


            #######################################################
            # Delay so the loop doesn't run too fast
            rospy.sleep(0.2)

if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    # Loop forever while processing callbacks
    cave_explorer.main_loop()



