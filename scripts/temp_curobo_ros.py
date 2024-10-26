import argparse
import os
import rclpy
from rclpy.node import Node
import numpy as np
import threading

# Import ROS2 message types
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from enum import Enum
from target_client.srv import TargetPose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CameraInfo
# Import curobo utilities (update these imports as needed)
from curobo.util_file import get_robot_configs_path, get_assets_path
# Import your CuroboMotionPlanner class
from temp_CuroboMotionPlanner import CuroboMotionPlanner
from fetch_camera import CameraLoader
from std_msgs.msg import Float32MultiArray

from curobo.types.math import Pose
from curobo.types.state import JointState

import struct

#type of motion received: 
class MotionType(Enum):
    FREESPACE = 0
    BOX_IN = 1
    BOX_OUT = 2
    SHELF_IN = 3
    SHELF_OUT = 4

# ROS2 SERVICE GIVES REQUEST WITH:
# DESIRED END POSE
# TYPE OF END GOAL (FREE, BOX_IN, BOX_OUT, SHELF_IN, SHELF_OUT)

# WHEN FREESPACE REQUEST, AN INTERMEDIARY POSE IS COMPUTED
# FOUND BY TRANSLATING POINT IN 3D SPACE ALONG GRASP VECTOR
# ALSO 
# Z-AXIS OF GRIPPER PROBABLY
class CuroboTrajectoryNode(Node):
    def __init__(self, cfg):
        super().__init__('curobo_trajectory_node')

        self.nvblox = os.getenv("NVBLOX", False)

        # Initialize CuroboMotionPlanner
        self.curoboMotion = CuroboMotionPlanner(cfg)

        if self.nvblox:
            self.curoboMotion.setup_motion_planner_with_collision_avoidance()
        else:
            self.curoboMotion.setup_motion_planner()  # Warmup happens here

        # Load the world and robot configurations for ISAAC SIM
        self.world_cfg = self.curoboMotion.world_cfg
        self.robot_cfg = self.curoboMotion.robot_cfg
        self.j_names = self.curoboMotion.j_names
        self.latest_joint_state = None
        self.start_js = None

        # Create the service
        self.target_srv = self.create_service(TargetPose, 'target_pose', self.target_pose_callback)

        # Subscriber for joint state messages
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
        
        # Publisher for joint trajectory messages
        # self.trajectory_publisher = self.create_publisher(
        #     JointTrajectory,
        #     'joint_trajectory_sim',+
        #     10)
        #         # Publisher for joint trajectory messages
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10)
        
        if self.nvblox:
            self.voxel_pub = self.create_publisher(Float32MultiArray, 'voxel_array', 10)
                # Subscriber for gripper status messages
        self.gripper_status_subscriber = self.create_subscription(
            Bool,
            'suction_status',
            self.gripper_status_callback,
            10)
        
        if self.nvblox:
            ## TODO: Replace continous subsription with invoked subscription for lower network load
            self.cam = CameraLoader()
            self.create_subscription( Image, '/head_camera/rgb/image_raw', self.cam.camera_callback)
            self.create_subscription( Image, '/head_camera/depth/image_rect_raw', self.cam.depth_callback)
            # self.create_subscription( Pose, '/camera/pose', self.pose_callback)
            self.cam.set_fetch_camera_pose()
            self.create_subscription( CameraInfo, '/head_camera/rgb/camera_info', self.cam.intrinsics_callback)

        self.gripper_status = None

        self.lock = threading.Lock()
        self.published_trajectory = None

        self.get_logger().info('Curobo Trajectory Node has been started.')
    
    def gripper_status_callback(self, msg):
        if self.gripper_status != msg.data:
            print(f"Changed gripper status: {msg.data}")
            self.gripper_status = msg.data

        
    def joint_state_callback(self, msg):
        # Update the latest joint state
        self.latest_joint_state = msg
        self.get_logger().debug('Updated latest joint state.')

    def get_current_joint_positions(self):
        if self.latest_joint_state is None:
            self.get_logger().error('No joint state available yet.')
            return None

        joint_positions = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position
        ))
        try:
            initial_js = [joint_positions[joint_name] for joint_name in self.curoboMotion.j_names]
            print(self.curoboMotion.j_names)
        except KeyError as e:
            self.curoboMotion.j_names
            self.get_logger().error(f'Joint name {e} not found in joint states.')
            return None
        
        return initial_js

    # GENERATE MOTION PLAN USING PREDEFINED MOTION PRIMS REQUESTED FROM TARGET POSE
    # In TargetPose.srv, we have target_pose and traj_type
    # traj_type is type of task to be performed
    # 1. GO TO HOVER POSITION ABOVE BOX
    # 2. APPROACH OBJECT BY GOING DOWN
    # 3. PICK OBJECT AND HOVER OVER BOX
    # 4. GO TO SHELF POSITION (NOT INSIDE ONLY SAME Z
    # 5. GO IN/OUT OF SHELF
        # POSE CONSTRAINT WITH RESPECT TO GRIPPER FRAME NOT WORLD
    
    def target_pose_callback(self, request, response):
        response.success = False
        self.cam.set_fetch_camera_pose() # UPDATE CURRENT CAMERA POSE wrt to BASE FRAME IN CAMERA DATA
        
        request_pose = request.target_pose
        request_type = MotionType(request.traj_type.data)
        print(request_type)
        print(f"Request pose: {request_pose}")
        initial_js = self.get_current_joint_positions()

        if request_pose is None and initial_js is None:
            self.get_logger().error('Cannot generate trajectory without current joint positions.')
            response.success = False
            self.published_trajectory = None
            return response
            
        target_pose = self.convert_pose_to_target_format(request_pose)
        
        if request_type == MotionType.FREESPACE:
            self.update_collision_world()    # ONLY UPDATE WHEN GOING TO INTERMEDIARY WAYPOINT.
            offset = -0.10 # -Z offset w.r.t ee goal frame
            self.curoboMotion.release_constraint()
        elif request_type == MotionType.BOX_OUT:
            offset = 0.30
            self.curoboMotion.set_constraint()
        elif request_type == MotionType.SHELF_OUT:
            offset = 0.20
            self.curoboMotion.set_constraint()
        elif request_type == MotionType.BOX_IN or request_type == MotionType.SHELF_IN:
            self.update_collision_world()
            offset = 0.0
            self.curoboMotion.set_constraint()
        else:
            self.get_logger().error('Invalid trajectory type.')
            response.success = False
            self.published_trajectory = None
            return response

        target_pose = self.curoboMotion.compute_intermediary_pose(target_pose, offset)

        trajectory = self.curoboMotion.generate_trajectory(
            initial_js=initial_js,
            goal_ee_pose=target_pose,
            # suction_status=self.gripper_status
        )
        if trajectory is None:
            self.get_logger().error('Failed to generate trajectory.')
            response.success = False
            self.published_trajectory = None
            return response
        
        joint_trajectory_msg = self.create_joint_trajectory_message(trajectory)
        self.published_trajectory = joint_trajectory_msg
        self.trajectory_publisher.publish(joint_trajectory_msg)
        self.start_js = initial_js

        self.get_logger().info('Published joint trajectory.')
        response.success = True
        self.get_logger().info("Response success: True")

        return response

    def update_collision_world(self):
        self.camera_data = self.cam.get_data()
        if self.camera_data is None:
            raise ValueError('Camera data not available.')
        # add in robot planning base frame - should be a constant or can get from tf
        camera_position = [0,0,0]
        camera_orientation = [0,0,0,1]
        self.camera_pose = Pose(
                position=self.tensor_args.to_device(camera_position),
                quaternion=self.tensor_args.to_device(camera_orientation),
            )
        voxels = self.curoboMotion.update_blox_from_camera(self.camera_data)
        self.publish_voxel_array(voxels.cpu().numpy())
        
    def publish_voxel_array(self, voxels):
        # Ensure voxels is in (N, 3) format for [x, y, z] coordinates
        if voxels.shape[1] < 3:
            self.get_logger().error("Voxel data must be in (N, 3) format.")
            return

        # Flatten the voxel data into a single list [x1, y1, z1, x2, y2, z2, ...]
        flat_voxel_data = voxels[:, :3].flatten().astype(np.float32).tolist()

        # Create and populate the Float32MultiArray message
        voxel_array_msg = Float32MultiArray()
        voxel_array_msg.data = flat_voxel_data

        # Publish the Float32MultiArray message
        self.voxel_pub.publish(voxel_array_msg)
        self.get_logger().info(f"Published {voxels.shape[0]} voxels as Float32MultiArray.")

    @staticmethod
    def convert_pose_to_target_format(pose_msg):
        # Extract the pose from PoseStamped
        pose = pose_msg.pose

        # Convert ROS2 Pose to the format expected by CuroboMotionPlanner
        # pose_curobo = [pose.position.x, pose.position.y, pose.position.z,
        #                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] 

        # HARD CODING ORIENTATION FOR BOX PICKING BECAUSE ITS A GIVEN --> BASED ON SUCTION_STATUS
        pose_curobo = [pose.position.x, pose.position.y, pose.position.z,
                       0.0, 0.707, 0.0, -0.707] 
        
        return pose_curobo

    def create_joint_trajectory_message(self, trajectory):
        # Create a JointTrajectory message
        joint_trajectory_msg = JointTrajectory()

        # Set joint names
        joint_trajectory_msg.joint_names = self.curoboMotion.j_names

        # Retrieve trajectory data
        positions_list = trajectory.get('positions', [])
        velocities_list = trajectory.get('velocities', [])
        accelerations_list = trajectory.get('accelerations', [])
        interpolation_dt = trajectory.get('interpolation_dt', 0.1)

        # Initialize time_from_start
        time_from_start = 0.0

        # Number of trajectory points
        num_points = len(positions_list)

        for idx in range(num_points):
            traj_point = JointTrajectoryPoint()

            # Handle positions
            positions = positions_list[idx]
            velocities = velocities_list[idx]
            accelerations = accelerations_list[idx]

            positions = [float(p) for p in positions]
            traj_point.positions = positions
            traj_point.velocities = velocities
            traj_point.accelerations = accelerations

            # Set time_from_start
            time_from_start += interpolation_dt
            traj_point.time_from_start = rclpy.duration.Duration(seconds=time_from_start).to_msg()

            # Append the trajectory point to the message
            joint_trajectory_msg.points.append(traj_point)

        return joint_trajectory_msg

if __name__ == "__main__":
    rclpy.init()
    cfg = "fetch.yml"
    curobo_node = CuroboTrajectoryNode(cfg)
    try:
        rclpy.spin(curobo_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure proper shutdown
        curobo_node.destroy_node()
        rclpy.shutdown()
