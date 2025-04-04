#!/usr/bin/env python3
import rospy
import numpy as np
import easyocr
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge, CvBridgeError 
import message_filters
import tf
from std_msgs.msg import Int32MultiArray

"""This node is used to get the coordinates of the box that shows the number repeated the least number of times.
Subscribes to the /percep/numberData topic to get the array of frequencies - get the non-zero lowest value's index
From the camera feed, match the box that has the same number - convert them into map/odom frame and take the average if multiple of the same number can be seen
Use depth to ignore the far boxes so that we don't scan the boxes from the exploration area"""

class BoxCoordinates():
    def __init__(self):
        self.bridge = CvBridge()
        self.processing = False

        self.depth_thresh = 100
        self.listener = tf.TransformListener()
        self.camera_info = rospy.wait_for_message("/front/rgb/camera_info", CameraInfo)
        self.img_frame = self.camera_info.header.frame_id

        self.ocr_detector = easyocr.Reader(["en"], gpu=True)

        self.img_curr = None
        self.depth_curr = None
        self.rgb_sub = message_filters.Subscriber("/front/rgb/image_raw", Image)
        self.depth_sub = message_filters.Subscriber('/front/depth/image_raw', Image)

        self.number = None
        self.coords = []
        self.num_freq_sub = rospy.Subscriber("/percep/numberData", Int32MultiArray, self.get_number)

        self.ats = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.01)
        self.ats.registerCallback(self.synced_images_callback) 

        #publisher
        self.box_coord_pub = rospy.Publisher("/percep/box_coord", PoseStamped, queue_size=10, latch=True)

    def get_number(self, msg):
        freq_array = np.array(msg.data)
        self.number = np.where(freq_array > 0)[0][np.argmin(freq_array[freq_array > 0])]
        return

    def synced_images_callback(self, rgb_data, depth_data):
        if self.processing:
            return
        
        self.processing = True

        try:
            rospy.loginfo(f"Received synced images: RGB shape={rgb_data.height}x{rgb_data.width}, encoding={rgb_data.encoding}")
            self.img_stamp = rgb_data.header.stamp

            print("Converting ROS image to OpenCV image")
            self.img_curr = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")  # Changed to "bgr8" for consistency
            self.depth_curr = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")
            print(f"OpenCV image shape: {self.img_curr.shape}, Depth image shape: {self.depth_curr.shape}")

            self.get_coords()
        
        except CvBridgeError as e:
            print(f"CvBridge error: {e}")
        
        finally:
            self.processing = False
    
    def get_coords(self):
        depth_thresh = self.depth_thresh

        if self.img_curr is None:
                rospy.logwarn("No image detected for processing..")
                return
        if self.number is None:
            rospy.logwarn("Number not received yet from /percep/numberData. Skipping frame.")
            return

        
        rospy.loginfo("Image detected, using ocr to detect numbers.....")
        result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
        rospy.loginfo("Numbers detected, proceeding to filter and transfrom.....")
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for detection in result:
            if len(detection[1]) > 1:  # not a single digit
                        continue
            if detection[1] == str(self.number): 
                if detection[2] < 0.99:
                        continue
                center = [(x + y) / 2 for x, y in zip(detection[0][0], detection[0][2])]
                x_center, y_center = int(center[0]), int(center[1])

                if 0 <= y_center < self.depth_curr.shape[0] and 0 <= x_center < self.depth_curr.shape[1]:
                    depth = self.depth_curr[y_center, x_center]
                    if not np.isfinite(depth) or depth <= 0 or depth > depth_thresh:
                        rospy.logwarn(f"Invalid or out-of-range depth at ({x_center}, {y_center}): {depth}")
                        continue
                
                rospy.loginfo("Close number detected, proceeding to transform coordinates and perform checks!!!!")
                X = (x_center - cx) * depth / fx
                Y = (y_center - cy) * depth / fy
                Z = depth

                transformed_pose = None
                p_in_cam = PoseStamped()
                p_in_cam.header.frame_id = self.img_frame
                p_in_cam.pose.position.x = X
                p_in_cam.pose.position.y = Y
                p_in_cam.pose.position.z = Z
                p_in_cam.pose.orientation.w = 1

                try:
                    self.listener.waitForTransform("map", self.img_frame, self.img_stamp, rospy.Duration(1.0))
                    transformed_pose = self.listener.transformPose("map", p_in_cam)
                    rospy.loginfo("Transformed to 'map' frame")

                except tf.Exception as e_map:
                    # rospy.logwarn(f"TF error transforming to 'map': {e_map}")

                    try:
                        self.listener.waitForTransform("odom", self.img_frame, self.img_stamp, rospy.Duration(1))
                        transformed_pose = self.listener.transformPose("odom", p_in_cam)
                        rospy.loginfo("Transformed to 'odom' frame")
                    
                    except tf.Exception as e_odom:
                        rospy.logerr(f"TF error transforming to both 'map' and 'odom'. Skipping this detection.")
                        continue
                
                X,Y,Z = transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z

                self.coords.append((X,Y,Z))
        
        rospy.loginfo(f"The coordinates of {self.number} detected till now are: {self.coords}")

        if self.coords:
            final_x, final_y, final_z = np.array(self.coords).mean(axis=0)
        else:
            rospy.logwarn("No valid coordinates detected to publish.")
            return

        final_coords = PoseStamped()
        final_coords.header.frame_id = transformed_pose.header.frame_id
        final_coords.pose.position.x = final_x
        final_coords.pose.position.y = final_y
        final_coords.pose.position.z = final_z
        final_coords.pose.orientation.w = 1

        self.box_coord_pub.publish(final_coords)
        rospy.loginfo(f"Published average coordinates for number {self.number}: ({final_x:.2f}, {final_y:.2f}, {final_z:.2f})")


        return
    
if __name__ == '__main__':
     try:
        rospy.init_node('box_coords')
        v = BoxCoordinates()
        rospy.spin()
     except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down visual node.")

        

                
                
            