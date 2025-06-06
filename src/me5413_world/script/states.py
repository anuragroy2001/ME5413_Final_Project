#!/usr/bin/env python3

from dynamic_reconfigure.client import Client as DynamicReconfigureClient
import rospy
import smach
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String,Bool,Int32MultiArray
from perception_depth import Image_segmentation
from threading import Thread
from cone_perception import ConeDetection
import tf
from box_coordinates import BoxCoordinates
import subprocess
import rosnode


class Jackal_Robot():
    def __init__(self):
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "map"
        self.goal = MoveBaseGoal()
        self.client=actionlib.SimpleActionClient('move_base',MoveBaseAction)
        self.timeout = rospy.Duration(240)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    
    def send_goal(self, goal):
        self.target_pose.pose.position.x = goal[0]
        self.target_pose.pose.position.y = goal[1]
        self.target_pose.pose.position.z = goal[2]
        self.target_pose.pose.orientation.x = goal[3]
        self.target_pose.pose.orientation.y = goal[4]
        self.target_pose.pose.orientation.z = goal[5]
        self.target_pose.pose.orientation.w = goal[6]
        self.goal.target_pose = self.target_pose
        self.client.send_goal(self.goal)
    
    def wait_for_server(self):
        if not self.client.wait_for_server(timeout=rospy.Duration(5)):
            rospy.logerr("Failed to connect to move_base server")
            return False
        return True
    
    def check_state(self,last_state,terminate_token=False):
        # Check current state
        current_state = self.client.get_state()

        #Update log when state changes
        #if current_state != last_state:
            #rospy.loginfo("Navigation state: %s", actionlib.GoalStatus.to_string(current_state))

        # Handle final states
        if current_state == actionlib.GoalStatus.SUCCEEDED:
            #rospy.loginfo("Goal reached!")
            terminate_token=True
            current_state='goal_reached'
        elif current_state in [actionlib.GoalStatus.PREEMPTED, actionlib.GoalStatus.ABORTED,
                                actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.RECALLED]:
            #rospy.loginfo("Navigation failed with state: %s", actionlib.GoalStatus.to_string(current_state))
            terminate_token=True
            current_state='failed'
        return current_state,terminate_token

    def check_timeout(self,start_time):
        # Check timeout
        if (rospy.Time.now() - start_time) > self.timeout:
            rospy.logwarn("Navigation timed out")
            self.client.cancel_goal()
            return True
        return False
    
    def move_forward(self,vel_x=1):
        # Create a Twist message to move the robot forward
        twist = Twist()
        twist.linear.x = vel_x
        twist.angular.z = 0.0
        # Publish the Twist message to the /cmd_vel topic
        self.cmd_vel_pub.publish(twist)
    
        

class Task1_obs_avoid_nav(smach.State):
    def __init__(self,goal=[19, -21, 0.0, 0.0, 0.0, -1.0, 0.0]):
        smach.State.__init__(self,outcomes=['goal_reached','failed','stopped'])
        self.timeout = rospy.Duration(240)
        self.goal=goal
        self.robot=Jackal_Robot()

    def execute(self,userdata):
        rospy.loginfo("Executing Task1_obs_avoid_nav")
        try:
            #Check if the move_base server is available
            if not self.robot.wait_for_server():
                return 'failed'
            
            #Initialize goal
            self.robot.send_goal(self.goal)

            start_time = rospy.Time.now()
            last_state=None

            while not rospy.is_shutdown():
                last_state,terminate_token=self.robot.check_state(last_state)
                if terminate_token:
                    return last_state

                # Check timeout
                if self.robot.check_timeout(start_time):
                    return 'failed'

                rospy.sleep(0.01)
        # If exit while loop due to shutdown
        except rospy.ROSInterruptException:
            rospy.logerr("ROS Interrupt Exception")
            self.robot.client.cancel_all_goals()
            return 'stopped'

class Task2_exploration(smach.State):
    def __init__(self,waypoint_list):
        smach.State.__init__(self,outcomes=['goal_reached','failed','stopped'])
        self.robot=Jackal_Robot()
        self.waypoint_list=waypoint_list
        self.start_waypoint=[19.0, -21, 0.0, 0.0, 0.0, -1.0, 0.0]
        self.waypoint_num=1
    
    def execute(self,userdata):
        rospy.loginfo("Executing Task2_exploration")
        try:
            img_seg=Image_segmentation()
            cone_det=ConeDetection()

            if not self.robot.wait_for_server():
                return 'failed'
            waypoint=self.start_waypoint
            while waypoint[1]<-3:
                self.robot.send_goal(waypoint)
                start_time = rospy.Time.now()
                last_state=None

                while not rospy.is_shutdown():
                    last_state,terminate_token=self.robot.check_state(last_state)
                    if terminate_token:
                        if last_state=='goal_reached':
                            break

                    # Check timeout
                    if self.robot.check_timeout(start_time):
                        return 'failed'

                    rospy.sleep(0.01)
                if waypoint[0] == 19.0:
                    waypoint[0] = 9.8
                else:
                    waypoint[0] = 19.0
                waypoint[1] += 1.5
                self.waypoint_num+=1
            
            return 'goal_reached'


        except rospy.ROSInterruptException:
            # If we exit while loop due to shutdown
            rospy.logerr("ROS Interrupt Exception")
            self.robot.client.cancel_all_goals()
            return 'stopped'

class Task3_move_to_bridge(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['goal_reached', 'failed','stopped'],
                             output_keys=['received_data'])  
        self.data_received = None
        self.subscriber = rospy.Subscriber("/percep/cone_estimated_position", PointStamped, self.callback)
        self.robot=Jackal_Robot()
        self.cone_coor=None
        
    def callback(self, msg):
        """Callback for the subscriber"""
        self.data_received = msg

    def extract_msg(self,msg):
        return [10, msg.point.y + 0.1,0,0,0,-1,0]
        

    def execute(self,userdata):
        try:
            start_time = rospy.Time.now()
            last_state=None
            while not rospy.is_shutdown():
                if self.data_received is not None and self.cone_coor is None:
                    self.cone_coor=self.extract_msg(self.data_received)
                    rospy.loginfo("Cone coordinates: %s", self.cone_coor)
                    if not self.robot.wait_for_server():
                        return 'failed'
                    else:
                        self.robot.send_goal(self.cone_coor)
                        #rospy.loginfo("Sent goal to move_base")

                last_state,terminate_token=self.robot.check_state(last_state)
                if terminate_token:
                    return last_state

                # Check timeout
                if self.robot.check_timeout(start_time):
                    return 'failed'
        except rospy.ROSInterruptException:
            # If we exit while loop due to shutdown
            rospy.logerr("ROS Interrupt Exception")
            self.robot.client.cancel_all_goals()
            return 'stopped'

class Task4_unlock_bridge(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['done', 'failed'])
        self.publisher=rospy.Publisher('/cmd_open_bridge',Bool,queue_size=10)
        self.ros_msg=Bool(True)
        self.timeout = 20  # seconds
        self.robot=Jackal_Robot()
        self.tf_listener = tf.TransformListener()
        self.current_pose=None
        self.connection_retries = 3  # number of retry attempts
    
    def get_current_x_position(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
            return trans[0]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Could not get robot pose: %s", e)
            return None

    def wait_for_connection(self):
            for attempt in range(self.connection_retries):
                start_time = rospy.Time.now()
                while (rospy.Time.now() - start_time).to_sec() < self.timeout/self.connection_retries:
                    if self.publisher.get_num_connections() > 0:
                        #rospy.loginfo("Connected to /cmd_open_bridge subscribers")
                        return True
                    #rospy.loginfo("Waiting for connection to /cmd_open_bridge subscribers...")
                    rospy.sleep(1)
                #rospy.logwarn(f"Connection attempt {attempt+1} failed")
            return False
    
    def execute(self,userdata):
        if not self.wait_for_connection():
            rospy.logerr("Failed to connect to subscribers for topic /cmd_open_bridge")
            return 'failed'
        try:
            rospy.loginfo("Executing Task4_pass_bridge")
            while self.current_pose is None or self.current_pose > 8:
                self.robot.move_forward(vel_x=0.5)
                self.current_pose=self.get_current_x_position()
                rospy.sleep(0.1)
            rospy.loginfo("Robot has moved to front of cone")
            self.publisher.publish(self.ros_msg)
            rospy.loginfo("Published message to topic:/cmd_open_bridge")
            rospy.sleep(0.05)  # Wait for a moment to ensure the message is received
            while self.current_pose is None or self.current_pose > 4.5:
                self.robot.move_forward(vel_x=0.7)
                self.current_pose=self.get_current_x_position()
                rospy.sleep(0.1)
            rospy.loginfo("Robot has passed the bridge")
            return 'done'
        except Exception as e:
            rospy.logerr("Failed to execute Task4: %s", str(e))
            return 'failed'

class Task5_choose_box(smach.State):
    def __init__(self,box_waypoint_list):
        smach.State.__init__(self,outcomes=['goal_reached','stopped','failed'])
        self.robot=Jackal_Robot()
        self.waypoint_list=box_waypoint_list
        self.timeout = 10  # seconds

        self.box_coord_sub=rospy.Subscriber("/percep/box_coord", PoseStamped,self.box_coordinates_callback)
        self.box_y_coordinate=None
        
    def box_coordinates_callback(self, msg):
        if self.box_y_coordinate is None:
            self.box_y_coordinate = msg.pose.position.y
            #rospy.loginfo("Received box coordinates: %s", self.box_y_coordinate)

    
    
    def execute(self,userdata):
        box_coord_script=BoxCoordinates()
        try:
            for waypoint in self.waypoint_list:
                self.robot.send_goal(waypoint)
                last_state=None
                while not rospy.is_shutdown():
                    last_state,_=self.robot.check_state(last_state)
                    rospy.sleep(0.1)
                    if last_state=='goal_reached':
                        #rospy.loginfo("goal_reached")
                        rospy.sleep(2)
                        break
                if self.box_y_coordinate is not None:
                    self.robot.send_goal([1.2,self.box_y_coordinate,0,0,0,0,1])
                    break
                else:
                    continue
            while not rospy.is_shutdown():
                terminate_token=False
                last_state,terminate_token=self.robot.check_state(last_state)
                rospy.sleep(0.5)
                if terminate_token:
                    return last_state
                
        except Exception as e:
            return 'stopped' 

# class Task5_change_map(smach.State):
#     def __init__(self):            
#         smach.State.__init__(self,outcomes=['map_changed','stopped','failed'])
#         self.robot=Jackal_Robot()
#         self.timeout = 10  # seconds
#         self.tf_listener = tf.TransformListener()
#         self.pose=None
        
#     def get_current_pose(self):
#         try:
#             (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))

#         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
#             rospy.logerr("Could not get robot pose: %s", e)
#             return None
        
    
#     def execute(self,userdata):
#         try:
#             rospy.loginfo("Switching maps...")
#             if '/map_server' in rosnode.get_node_names():
#                 rospy.loginfo("Killing old map_server...")
#                 subprocess.call(['rosnode', 'kill', '/map_server'])
#                 rospy.sleep(1)
#             rospy.loginfo("Launching new map: %s", self.new_map_launch)
#             self.process = subprocess.Popen(['roslaunch', self.new_map_launch])
#             rospy.sleep(3.0)  # Wait for the map to load   

#         except Exception as e:
#             pass




if __name__=="__main__":        

    pass 
