import argparse
import os
import rclpy
from rclpy.node import Node
import numpy as np
import threading
# Import ROS2 message types
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from target_client.srv import TargetPose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
# Import curobo utilities (update these imports as needed)
from curobo.util_file import get_robot_configs_path, get_assets_path
# Import your CuroboMotionPlanner class
from CuroboMotionPlanner import CuroboMotionPlanner
from curobo.types.state import JointState as JointStateC

class CuroboTrajectoryNode(Node):
    def __init__(self, cfg, shared_data):
        super().__init__('curobo_trajectory_node')
        self.shared_data = shared_data
        # Initialize CuroboMotionPlanner
        self.curoboMotion = CuroboMotionPlanner(cfg)
        self.curoboMotion.setup_motion_planner()  # Warmup happens here

        # Load the world and robot configurations for ISAAC SIM
        self.world_cfg = self.curoboMotion.world_cfg
        self.robot_cfg = self.curoboMotion.robot_cfg
        self.j_names = self.curoboMotion.j_names
        self.latest_joint_state = None
        self.start_js = None
        self.shared_data.world_cfg = self.world_cfg
        self.shared_data.robot_cfg = self.robot_cfg
        self.shared_data.j_names = self.j_names
        # Create the service
        self.target_srv = self.create_service(TargetPose, 'target_pose', self.target_pose_callback)

        # Subscriber for joint state messages
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
        
        # Publisher for joint trajectory messages
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10)
                # Subscriber for gripper status messages
        self.gripper_status_subscriber = self.create_subscription(
            Bool,
            '/suction_status',
            self.gripper_status_callback,
            10)
        self.gripper_status = None

        self.lock = threading.Lock()
        self.published_trajectory = None

        self.get_logger().info('Curobo Trajectory Node has been started.')
    def gripper_status_callback(self, msg):
        self.gripper_status = msg.data
        self.shared_data.gripper_status = msg.data
        
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
            self.shared_data.start_js = initial_js
        except KeyError as e:
            self.curoboMotion.j_names
            self.get_logger().error(f'Joint name {e} not found in joint states.')
            return None
        
        return initial_js

    def target_pose_callback(self, request, response):
        response.success = False
        request_pose = request.target_pose
        initial_js = self.get_current_joint_positions()

        if request_pose is None and initial_js is None:
            self.get_logger().error('Cannot generate trajectory without current joint positions.')
            response.success = False
            self.published_trajectory = None
            return response
            
        target_pose = self.convert_pose_to_target_format(request_pose)

        trajectory = self.curoboMotion.generate_trajectory(
            initial_js=initial_js,
            goal_ee_pose=target_pose)

        if trajectory.get('success') is False:
            self.get_logger().error('Failed to generate trajectory.')
            response.success = False
            self.published_trajectory = None
            return response
        
        joint_trajectory_msg = self.create_joint_trajectory_message(trajectory)
        self.published_trajectory = joint_trajectory_msg
        self.trajectory_publisher.publish(joint_trajectory_msg)
        self.shared_data.published_trajectory = joint_trajectory_msg
        self.start_js = initial_js

        self.get_logger().info('Published joint trajectory.')
        response.success = True
        return response

    @staticmethod
    def convert_pose_to_target_format(pose_msg):
        # Extract the pose from PoseStamped
        pose = pose_msg.pose

        # Convert ROS2 Pose to the format expected by CuroboMotionPlanner
        pose_curobo = [pose.position.x, pose.position.y, pose.position.z,
                       pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] 
        
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

