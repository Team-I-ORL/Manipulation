import os
import argparse

import rclpy
from rclpy.node import Node
import numpy as np
# Import ROS2 message types
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# Import curobo utilities (update these imports as needed)
from curobo.util_file import get_robot_configs_path, get_assets_path

# Import your CuroboMotionPlanner class
from CuroboMotionPlanner import CuroboMotionPlanner

class CuroboTrajectoryNode(Node):
    def __init__(self, cfg):
        super().__init__('curobo_trajectory_node')

        # Initialize CuroboMotionPlanner
        self.curoboMotion = CuroboMotionPlanner(cfg)
        self.curoboMotion.setup_motion_planner()  # Warmup happens here
        self.latest_joint_state = None
        self.pending_pose = None   
        # Publishers and Subscribers
        # Subscriber for goal pose messages
        self.pose_subscriber = self.create_subscription(
            Pose,
            'desired_pose',
            self.pose_callback,
            10)
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

        self.get_logger().info('Curobo Trajectory Node has been started.')

    def joint_state_callback(self, msg):
        # Update the latest joint state
        self.latest_joint_state = msg
        self.get_logger().debug('Updated latest joint state.')

        # If there is a pending desired pose, process it now
        if self.pending_pose is not None:
            self.get_logger().info('Processing pending desired pose.')
            initial_js = self.get_current_joint_positions()
            if initial_js is None:
                self.get_logger().error('Cannot generate trajectory without current joint positions.')
                return

            self.process_pose(self.pending_pose, initial_js)
            # Clear the pending pose
            self.pending_pose = None

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
        except KeyError as e:
            self.get_logger().error(f'Joint name {e} not found in joint states.')
            return None

        return initial_js


    def pose_callback(self, msg):
        self.get_logger().info('Received desired pose.')

        if self.latest_joint_state is None:
            self.get_logger().warn('No joint state available yet. Storing desired pose for later processing.')
            self.pending_pose = msg
            return

        # Extract current joint positions
        initial_js = self.get_current_joint_positions()
        if initial_js is None:
            self.get_logger().error('Cannot generate trajectory without current joint positions.')
            return

        # Process the pose
        self.process_pose(msg, initial_js)

    def process_pose(self, msg, initial_js):
        # Convert Pose message to the format expected by CuroboMotionPlanner
        target_pose = self.convert_pose_to_target_format(msg)

        # Generate trajectory for a single goal
        trajectory = self.curoboMotion.generate_trajectory(
            initial_js=initial_js,
            goal_ee_pose=target_pose)

        if trajectory is None:
            self.get_logger().error('Failed to generate trajectory.')
            return

        # Publish the trajectory
        joint_trajectory_msg = self.create_joint_trajectory_message(trajectory)
        self.trajectory_publisher.publish(joint_trajectory_msg)
        self.get_logger().info('Published joint trajectory.')

    @staticmethod
    def convert_pose_to_target_format(pose_msg):
        # Convert ROS2 Pose message to the format expected by curoboMotion
        pose_curobo = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
                       pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z] 
        
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

            positions = [float(p) for p in positions]
            traj_point.positions = positions

            # Set time_from_start
            time_from_start += interpolation_dt
            traj_point.time_from_start = rclpy.duration.Duration(seconds=time_from_start).to_msg()

            # Append the trajectory point to the message
            joint_trajectory_msg.points.append(traj_point)

        return joint_trajectory_msg

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="fetch.yml", 
        help="Configuration file to load"
    )
    parsed_args = parser.parse_args()

    # Initialize ROS2
    rclpy.init(args=args)

    # Create the node
    curobo_node = CuroboTrajectoryNode(parsed_args.cfg)

    # Spin the node (this provides the main loop)
    rclpy.spin(curobo_node)

    # Clean up
    curobo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()