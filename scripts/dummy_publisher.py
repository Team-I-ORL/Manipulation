#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Import ROS2 message types
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class DummyPublisherNode(Node):
    def __init__(self):
        super().__init__('dummy_publisher_node')

        # Create publishers
        self.pose_publisher = self.create_publisher(Pose, 'desired_pose', 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Joint names and positions
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint"
        ]

        self.retract_config = [
            1.3056849,
            1.4040100,
            -0.34258141,
            1.743283,
            0.017052,
            1.627947,
            -0.129718
        ]

        self.get_logger().info('DummyPublisherNode has been started.')

        # Publish joint states immediately
        self.publish_joint_states()

        # Set timer to publish joint states at fixed rate
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        # Schedule desired pose publishing after a short delay
        self.create_timer(0.2, self.publish_desired_pose)

    def publish_desired_pose(self):
        # Publish desired pose once
        desired_pose_msg = Pose()

        # Define desired pose (example values)
        desired_pose_msg.position.x = 0.735
        desired_pose_msg.position.y = 0.0
        desired_pose_msg.position.z = 0.6857

        # Orientation (quaternion)
        desired_pose_msg.orientation.x = 1.0
        desired_pose_msg.orientation.y = 0.0
        desired_pose_msg.orientation.z = 0.0
        desired_pose_msg.orientation.w = 0.0

        # Publish the desired pose
        self.pose_publisher.publish(desired_pose_msg)
        self.get_logger().info('Published desired pose once.')

    def publish_joint_states(self):
        # Publish joint states periodically
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()

        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.retract_config
        # Optionally, you can add velocities and efforts if needed
        joint_state_msg.velocity = []
        joint_state_msg.effort = []

        # Publish the joint states
        self.joint_state_publisher.publish(joint_state_msg)
        self.get_logger().debug('Published joint states.')

def main(args=None):
    rclpy.init(args=args)

    node = DummyPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful exit with Ctrl+C

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()