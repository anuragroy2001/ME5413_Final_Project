#!/usr/bin/env python3
import rospy
import tf
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
import message_filters
from std_msgs.msg import Int32MultiArray


class Image_segmentation:
    def __init__(self, rate=10):
        rospy.loginfo(f" Running rate: {rate}")
        self.bridge = CvBridge()
        self.rate = rospy.Rate(rate)
        self.detected_numbers_positions = {}
        self.processing = False

        self.depth_thresh = 100 #TODO: change the threshold values
        self.depth_thresh_close = 20 #TODO: change the threshold values
        self.distance_thresh = 7.0

        self.listener = tf.TransformListener()
        self.camera_info = rospy.wait_for_message("/front/rgb/camera_info", CameraInfo)
        self.img_frame = self.camera_info.header.frame_id

        # OCR configuration
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)

        self.img_curr = None
        self.depth_curr = None
        self.rgb_sub = message_filters.Subscriber("/front/rgb/image_raw", Image)
        self.depth_sub = message_filters.Subscriber('/front/depth/image_raw', Image)

        #Publisher
        self.number_database = rospy.Publisher("/percep/numberData", Int32MultiArray, queue_size=1, latch=True)
        
        self.ats = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.01)
        self.ats.registerCallback(self.synced_images_callback) 

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

            detected_numbers = self.ocr_detection_tranfrom()
            rospy.loginfo(f"The current number dictionary is this: {detected_numbers}")

        except CvBridgeError as e:
            print(f"CvBridge error: {e}")
        
        finally: 
             self.processing = False
    
    def ocr_detection_tranfrom(self): 
        """Detect all the numbers in the current frame
            Apply depth thresholding on all the BB and choose the close ones"""
        if self.img_curr is None:
                rospy.logwarn("No image detected for processing..")
                return
        rospy.loginfo("Image detected, using ocr to detect numbers.....")
        result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
        rospy.loginfo("Numbers detected, proceeding to filter and transfrom.....")
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for detection in result:
            # detection[0]: the bounding box of the detected text
            # detection[1]: the detected text
            # detection[2]: the confidence of the detected text
            if len(detection[1]) > 1:  # not a single digit
                    continue
            if detection[2] < 0.99:
                    continue
            center = [(x + y) / 2 for x, y in zip(detection[0][0], detection[0][2])]
            x_center, y_center = int(center[0]), int(center[1])

            if 0 <= y_center < self.depth_curr.shape[0] and 0 <= x_center < self.depth_curr.shape[1]:
                depth = self.depth_curr[y_center, x_center]
                if not np.isfinite(depth) or depth < self.depth_thresh_close or depth > self.depth_thresh:
                    rospy.logwarn(f"Invalid or out-of-range depth at ({x_center}, {y_center}): {depth}")
                    continue

            if detection[1] in {"0"}:
                continue

            rospy.loginfo("Close number detected, proceeding to transform coordinates and perform checks!!!!")
            X = (x_center - cx) * depth / fx
            Y = (y_center - cy) * depth / fy

            Z = depth - 1.0  # Subtract 1.0 to avoid collision with the target
            if Z < 0:
                Z = 0.0
            
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

            """Now that you have transformed pose - store it in dictionary along with the number!!"""
            rospy.loginfo("The odom position of this BB is  x={}, y={}, z={}".format(
            transformed_pose.pose.position.x,
            transformed_pose.pose.position.y,
            transformed_pose.pose.position.z))

            if detection[1].isdigit() and 0 <= int(detection[1]) <= 9:
                number = int(detection[1])
            else:
                continue
            
            if number not in self.detected_numbers_positions:
                self.detected_numbers_positions[number] = []

            already_seen = False
            for pos in self.detected_numbers_positions[number]:
                dist = np.linalg.norm([pos[0] - X, pos[1] - Y, pos[2] - Z])
                if dist < self.distance_thresh:
                    already_seen = True
                    break
            
            if already_seen:
                rospy.loginfo(f"Number:{number} at ({X:.2f}, {Y:.2f}, {Z:.2f}) already detected. Skipping.")
                continue
            
            self.detected_numbers_positions[number].append((X, Y, Z))
            rospy.loginfo(f"New: {number} detected at ({X:.2f}, {Y:.2f}, {Z:.2f}), processing...")

            self.num_freq_pub(self.detected_numbers_positions)

        return self.detected_numbers_positions
    
    def num_freq_pub(self, detected_numbers):
        result = [0]*10
        for key, value in detected_numbers.items():
            result[key] = len(value)
        number_msg = Int32MultiArray()
        number_msg.data = result

        self.number_database.publish(number_msg)
        return



    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()
        

if __name__ == '__main__':
    try:
        rospy.init_node('visual_perception')
        v = Image_segmentation()
        v.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down visual node.")

    
            

        
        
