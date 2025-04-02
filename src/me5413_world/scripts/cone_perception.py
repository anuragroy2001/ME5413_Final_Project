#!/usr/bin/env python3
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, PoseStamped
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError 
import message_filters
import numpy as np
"""
In this file we're trying to estimate the location of the cone based on all it's sighting during the exploration of the cubes area.

Will be using Hue based Template matching as robot moves around a lot and using the shape to identify it will be difficult.
Since it has a distinct orange color Hue based matching might work better.

As the robot moves around in the simulation environment, we keep track of the sightings of the cone - convert it into map/odom coordinates - weighted average them
out based on the confidence scores of the detections

If this doesn't work very well we can also try Kalman filter for the tracking of the cone position.
"""

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cone.png")

class ConeDetection():
    def __init__(self):
        self.bridge = CvBridge()
        self.processing = False
        self.template = cv2.imread(template_path)

        self.detected_positions = []
        self.confidence_scores = []
        self.weighted_position_sum = np.zeros(3)
        self.total_confidence = 0.0

        self.avg_pub = rospy.Publisher("/percep/cone_estimated_position", PointStamped, queue_size=10, latch=True)

        self.listener = tf.TransformListener()
        self.camera_info = rospy.wait_for_message("/front/rgb/camera_info", CameraInfo)
        self.img_frame = self.camera_info.header.frame_id

        if self.template is None:
            rospy.logerr(f"Failed to load template image from: {template_path}")
        else:
            rospy.loginfo(f"Successfully loaded template from: {template_path}")

        self.rgb_sub = message_filters.Subscriber("/front/rgb/image_raw", Image)
        self.depth_sub = message_filters.Subscriber('/front/depth/image_raw', Image)

        self.ats = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.01)
        self.ats.registerCallback(self.synced_images_callback)


    
    def synced_images_callback(self, rgb_data, depth_data):
        if self.processing:
            return

        self.processing = True
        print(f"Received synced images: RGB shape={rgb_data.height}x{rgb_data.width}, encoding={rgb_data.encoding}")
        
        try:
            print("Converting ROS image to OpenCV image")
            self.img_stamp = rgb_data.header.stamp

            self.img_curr = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")  # Changed to "bgr8" for consistency
            self.depth_curr = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")

            print(f"OpenCV image shape: {self.img_curr.shape}, Depth image shape: {self.depth_curr.shape}")
            self.match_template()

        except CvBridgeError as e:
            print(f"CvBridge error: {e}")
        
        finally: 
             self.processing = False

    
    def match_template(self):
        rospy.loginfo("Starting hue-based template matching...")

        image_hsv = cv2.cvtColor(self.img_curr, cv2.COLOR_BGR2HSV)
        template_hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)

        image_h = image_hsv[:, :, 0] # extracting only hue channel
        template_h = template_hsv[:, :, 0]

        original_height, original_width = template_h.shape[:2]
        scales = [0.25, 0.35, 0.40, 0.50, 0.60, 0.75, 0.85, 1.0]
        max_match_val = -1
        best_scale = None
        best_max_loc = None

        for scale in scales:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized_template = cv2.resize(template_h, (new_width, new_height), interpolation=cv2.INTER_AREA)
                                        
            result = cv2.matchTemplate(image_h, resized_template, cv2.TM_CCOEFF_NORMED) # TODO: Can try changing this model as well
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            rospy.loginfo(f"Scale {scale}: match value = {max_val:.3f}, location = {max_loc}")

            if max_val > max_match_val:
                max_match_val = max_val
                best_scale = scale
                best_max_loc = max_loc
        
        if max_match_val >= 0.50:
            rospy.loginfo(f"Cone detected in image!! Confidence = {max_match_val:.3f}")
            x, y = best_max_loc
            width = int(original_width * best_scale)
            height = int(original_height * best_scale)

            self.process_cone_detection(x, y, width, height, max_match_val)

            # Draw bounding box for debug TODO: comment out this part after debugging
            debug_image = self.img_curr.copy()
            cv2.rectangle(debug_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(debug_image, f"{max_match_val:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("BB of the cone", debug_image)
            cv2.waitKey(1)
            
    def process_cone_detection(self,x,y,w,h,confidence):
        try:
            u = x+w // 2
            v = y+h // 2
            depth = self.depth_curr[v,u]

            if depth == 0 or np.isnan(depth):
                rospy.logerr("Invalid or missing depth at center pixel.")
                return
            
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]

            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
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
                    return
            
            X,Y,Z = transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z
            rospy.loginfo(f"Cone's position is  X:{X}, Y:{Y}, Z:{Z} with a confidence score of: {confidence}")

            self.detected_positions.append((X,Y,Z))
            self.confidence_scores.append(confidence)

            if len(self.detected_positions) > 30: # if too many sightings this prevents storing too many positions
                self.detected_positions.pop(0)
                self.confidence_scores.pop(0)
            
            rospy.loginfo(f"Positions of last 30 cones detected till now: {self.detected_positions} & their confidence scores: {self.confidence_scores}")

            pos_np = np.array([X, Y, Z])
            self.weighted_position_sum += pos_np * confidence
            self.total_confidence += confidence

            avg_pos = self.weighted_position_sum / self.total_confidence
            rospy.loginfo(f"Rolling average cone position: {tuple(avg_pos)}")

            avg_point = PointStamped()
            avg_point.header.stamp = rospy.Time.now()
            avg_point.header.frame_id = transformed_pose.header.frame_id
            avg_point.point.x = avg_pos[0]
            avg_point.point.y = avg_pos[1]
            avg_point.point.z = avg_pos[2]

            self.avg_pub.publish(avg_point)
            rospy.loginfo("Average Position Published to cone_expected_position")

        except Exception as e:
            rospy.logwarn(f"[TF ERROR] Cone detection transform failed: {str(e)}")

if __name__ == '__main__':
    rospy.init_node("ConeDetection")
    D = ConeDetection()
    rospy.spin()