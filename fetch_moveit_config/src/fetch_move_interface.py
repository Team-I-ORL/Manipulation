#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib import SimpleActionClient
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandGoal
from sensor_msgs.msg import JointState
import numpy as np

class RobotArm:
    def __init__(self):
        # self.traj_client = SimpleActionClient("gripper_controller/gripper_action", GripperCommandAction)
        self.traj_client = SimpleActionClient("arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.traj_client.wait_for_server()

        self.desired_joint_values = []
        self.current_joint_values = []
        
        # Order is [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self.home_state = [1.59, 1.5, 0.0, 1.7784, 0.0, 1.18, -1.57]
        self.inter_pose_1 = [1.6056, 1.518, -0.8157, 1.7784, 0.0, 1.444, -2.3778]
        self.inter_pose_2 = [0.3637, 1.518, -2.8291, 1.7784, 0.0, 1.3724, -3.1415]

        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory.header.stamp = rospy.Time.now() #+ rospy.Duration(1.0)
        self.goal.trajectory.joint_names = self.joint_names
        
        self.desired_joint_value_sub = rospy.Subscriber("isaac_joint_commands", JointState, self.append_joint_values)
        
    def append_joint_values(self, data):

        print("Received Joint Command")
        pose_name = data.name[0]
        print(pose_name)
        if pose_name == "home":
            self.desired_joint_values = self.home_state
        
        elif pose_name == "inter_pose_1":
            self.desired_joint_values = self.inter_pose_1
        
        elif pose_name == "inter_pose_2":
            self.desired_joint_values = self.inter_pose_2
        
        else:
            # Extract the desired joint values
            self.desired_joint_values = data.position

            # Check if any 5 values are 0
            if sum(1 for v in self.desired_joint_values if v == 0.0) >= 5:
                print("Invalid Joint Pose - 5 values are 0")
                return
            
            # Check if all values are 0
            if all(v == 0.0 for v in self.desired_joint_values): 
                print("Invalid Joint Pose - All values are 0")
                return 

        move = FollowJointTrajectoryGoal()
        move.trajectory.header.stamp = rospy.Time.now() #+ rospy.Duration(2.0)
        move.trajectory.joint_names = self.joint_names
        point1 = JointTrajectoryPoint()
        point1.positions = self.desired_joint_values 
        point1.velocities = [0.0] * 7
        point1.time_from_start = rospy.Duration(1.0)
        move.trajectory.points.append(point1)
        self.traj_client.send_goal(move)
        print("Moving to Desired Joint Pose")


    def move_Arm(self):

        self.desired_joint_values = self.home_state

        move = FollowJointTrajectoryGoal()
        move.trajectory.header.stamp = rospy.Time.now() #+ rospy.Duration(2.0)
        move.trajectory.joint_names = self.joint_names
        point1 = JointTrajectoryPoint()
        point1.positions = self.desired_joint_values 
        point1.velocities = [0.0] * 7
        point1.time_from_start = rospy.Duration(1.0)
        move.trajectory.points.append(point1)
        self.traj_client.send_goal(move)
        print("Moving to Desired Joint Pose")

        return 
            

def main():
    rospy.init_node("fetch_move_interface")
    arm = RobotArm()

    arm.move_Arm()

    rospy.spin()


if __name__ == "__main__":
    main()
