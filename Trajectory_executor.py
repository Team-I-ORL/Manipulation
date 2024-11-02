import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from enum import Enum
from std_msgs.msg import Int32
class Status(Enum):
    PENDING         = 0
    ACTIVE          = 1
    PREEMPTED       = 2
    SUCCEEDED       = 3
    ABORTED         = 4
    REJECTED        = 5 
    PREEMPTING      = 6 
    RECALLING       = 7
    RECALLED        = 8
    LOST            = 9

class RobotArm:
    def __init__(self):
        # Initialize the action client for the follow_joint_trajectory action
        self.traj_client = actionlib.SimpleActionClient("arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

        rospy.loginfo("Waiting for the follow_joint_trajectory action server...")
        self.traj_client.wait_for_server(rospy.Duration(5.0))
        rospy.loginfo("Action server started, ready to send joint trajectory goals.")

        # Subscribe to the JointTrajectory topic
        self.joint_trajectory_sub = rospy.Subscriber("joint_trajectory", JointTrajectory, self.trajectory_callback)

        # Publisher for status of the robot arm
        self.status_pub = rospy.Publisher("robot_arm_status", Int32, queue_size=1)

        # Flag to indicate if the robot is currently executing a trajectory
        self.executing_trajectory = False

    def trajectory_callback(self, trajectory_msg):
        """
        Callback that triggers when a JointTrajectory message is received.
        Converts it to a FollowJointTrajectoryGoal and sends it to the action server.
        """
        print("Received trajecotry >>>>>>>>>> ", trajectory_msg)
        if self.executing_trajectory:
            rospy.loginfo("Currently executing a trajectory. Ignoring new trajectory.")
            return

        rospy.loginfo("Received new trajectory from topic.")

        # Create a FollowJointTrajectoryGoal from the received trajectory
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory_msg  # The received JointTrajectory message is directly assigned
        
        # Set the goal's timestamp (1 second from now)
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)
        
        joint_tolerance = JointTolerance()

        # Define goal tolerance for each joint
        tolerance_position = 0.01  # 1 cm deviation allowed in position
        tolerance_velocity = 0.01  # Small deviation in velocity allowed
        tolerance_acceleration = 0.01  # Small deviation in acceleration allowed
        
        
        for joint_name in goal.trajectory.joint_names:
            tolerance = JointTolerance()
            tolerance.name = joint_name
            tolerance.position = tolerance_position
            tolerance.velocity = tolerance_velocity
            tolerance.acceleration = tolerance_acceleration
            goal.goal_tolerance.append(tolerance)

        print("joint trajectory goal for action server ::::::::  ", goal)
        # Send the goal to the action server
        self.traj_client.send_goal(goal, feedback_cb=self.feedback_callback, done_cb=self.done_callback)
        self.executing_trajectory = True
        rospy.loginfo("Trajectory goal sent to action server.")

    def feedback_callback(self, feedback):
        """
        Callback to handle feedback from the action server.
        """
        print("State of Arm: ", self.traj_client.get_state())
        self.status_pub.publish(self.traj_client.get_state())
        # rospy.loginfo("Received feedback during execution of the trajectory.")

    def done_callback(self, state, result):
        """
        Callback to handle when the action is complete.
        """
        rospy.loginfo("Trajectory execution completed with state: %s, result: %s", state, result)
        # rospy.loginfo("Trajectory execution completed with state: ", state)

        # Publish the status of the robot arm
        self.status_pub.publish(state)

        # Reset the flag to indicate that the trajectory execution is complete
        self.executing_trajectory = False

    def get_state(self):
        """
        Returns the current state of the action.
        """
        return self.traj_client.get_state()


def main():
    # Initialize the ROS node
    rospy.init_node("robot_arm_trajectory_controller")

    # Create an instance of the RobotArm class
    arm = RobotArm()

    # Keep the node running
    rospy.spin()


if __name__ == "__main__":
    main()