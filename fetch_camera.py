# a class that has all function callbacks to get the camera data written above from topics being published on ros by the camera
#         rgb = torch.from_numpy(rgb_np)
#         rgba = torch.cat([rgb, torch.ones_like(rgb[:, :, 0:1]) * 255], dim=-1)
#         depth = torch.from_numpy(depth_np).float()
#         pose = torch.from_numpy(pose_np).float()
#         intrinsics = torch.from_numpy(intrinsics_np).float()

#         return {
#             "rgba": rgba,
#             "depth": depth,
#             "pose": pose,
#             "intrinsics": intrinsics,
#             "raw_rgb": rgb_np,
#             "raw_depth": depth_np,
#             "rgba_nvblox": rgba.permute((1, 2, 0)).contiguous(),
#         }

# it should have a get data funtion that return updated camera data in the above dict format gathered from different camera topics
# rgb, depth, pose, intrinsics, raw_rgb, raw_depth, rgba_nvblox

import torch
import rclpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf_transformations as tf
from rclpy.duration import Duration

class CameraLoader():
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.camera_data = {
            "rgba": None,
            "depth": None,
            "pose": None,
            "intrinsics": None,
            "raw_rgb": None,
            "raw_depth": None,
            "rgba_nvblox": None,
        }

    def rgb_callback(self, data):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
            rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()   
            self.camera_data["raw_rgb"] = rgb_tensor     
            rgba = torch.cat([rgb_tensor, torch.ones_like(rgb_tensor[:, :, 0:1]) * 255], dim=-1)
            self.camera_data["rgba"] = rgba
            # self.node.get_logger().info("RGBA image data updated in camera_data dictionary.")
        except Exception as e:
            self.node.get_logger().error(f"Failed to convert image: {e}")

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            depth_tensor = torch.from_numpy(depth_image/1000).float()
            # print("depth data : ", depth_tensor.mean())

            self.camera_data["depth"] = depth_tensor
            # self.node.get_logger().info("Depth image data updated in camera_data dictionary.")
        except Exception as e:
            self.node.get_logger().error(f"Failed to convert depth image: {e}")
    
    def pose_callback(self, data):
        # goes in as a float
        pose = torch.from_numpy(data).float()
        self.camera_data["pose"] = pose
    
    def set_fetch_camera_pose(self):
        while True:
            try:
                # Wait and lookup the transform from 'base_link' to 'camera_rgb_link' 
                transform = self.tf_buffer.lookup_transform( 'base_link', 'head_camera_depth_optical_frame', rclpy.time.Time(), timeout=Duration(seconds=3.0))
                print("transform : ", transform)
                # Extract the translation (position) data
                translation = transform.transform.translation
                position = torch.tensor([translation.x, translation.y, translation.z], dtype=torch.float32)

                # Extract the rotation (orientation) quaternion
                rotation = transform.transform.rotation
                quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
                
                # Convert quaternion to rotation matrix
                rotation_matrix = torch.tensor(tf.quaternion_matrix(quaternion)[:3, :3], dtype=torch.float32)
                quaternion = torch.tensor(quaternion, dtype=torch.float32)
                # Store the pose in the camera_data dictionary
                self.camera_data["pose"] = {
                    "position": position,
                    "orientation": quaternion[[3,0,1,2]]
                }
                self.node.get_logger().info("Camera pose fetched and stored in camera_data.")
                break

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.node.get_logger().error(f"Could not fetch transform: {e}")
                break


    def intrinsics_callback(self, data):
        try:
            intrinsics_matrix = torch.tensor(data.k, dtype=torch.float32).view(3, 3)
            self.camera_data["intrinsics"] = intrinsics_matrix
            # self.node.get_logger().info("Intrinsics data updated in camera_data dictionary.")
        except Exception as e:
            self.node.get_logger().error(f"Failed to process camera intrinsics: {e}")

    def get_data(self):
        # self.clip_camera(self.camera_data)
        return self.camera_data
    
    def clip_camera(self, camera_data):
    # clip camera image to bounding box:
        h_ratio = 0.05
        w_ratio = 0.05
        depth = camera_data["raw_depth"]
        depth_tensor = camera_data["depth"]
        h, w = depth_tensor.shape
        depth[: int(h_ratio * h), :] = 0.0
        depth[int((1 - h_ratio) * h) :, :] = 0.0
        depth[:, : int(w_ratio * w)] = 0.0
        depth[:, int((1 - w_ratio) * w) :] = 0.0

        depth_tensor[: int(h_ratio * h), :] = 0.0
        depth_tensor[int(1 - h_ratio * h) :, :] = 0.0
        depth_tensor[:, : int(w_ratio * w)] = 0.0
        depth_tensor[:, int(1 - w_ratio * w) :] = 0.0