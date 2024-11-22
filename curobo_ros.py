import argparse
import os
import rclpy
from rclpy.node import Node
import numpy as np
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

# Import ROS2 message types
from std_msgs.msg import Bool, Int32, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from enum import Enum
from orbiter_bt.srv import MoveArm
import time
from copy import deepcopy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image, CameraInfo
# Import curobo utilities (update these imports as needed)
from curobo.util_file import get_robot_configs_path, get_assets_path
# Import your CuroboMotionPlanner class
from CuroboMotionPlanner2 import CuroboMotionPlanner
from curobo.types.state import JointState as JointStateC
# from fetch_camera import CameraLoader
from std_msgs.msg import Float32MultiArray
from curobo.types.math import Pose as PoseC
from curobo.geom.types import Cuboid


#type of motion received: 
class MotionType(Enum):
    TORSO_DOWN = -2
    TORSO_UP = -1
    FREESPACE = 0
    BOX_IN = 1
    BOX_OUT = 2
    SHELF_IN = 3
    SHELF_OUT = 4
    RANDOM = 5
    APPROACH = 6
    HOME = 7
    RESTOCK_HOME = 8
    SHELF = 9
    BOX = 10
    FREESPACE_ARUCO = 11
    DROP_IN = 12

class Status(Enum):
    PENDING = 0
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    REJECTED = 5 
    PREEMPTING = 6 
    RECALLING = 7
    RECALLED = 8
    LOST = 9

class CuroboTrajectoryNode(Node):
    def __init__(self, cfg):
        super().__init__('curobo_trajectory_node')
        self.arm_callback_group = MutuallyExclusiveCallbackGroup()
        self.srv_callback_group = MutuallyExclusiveCallbackGroup()
        self.robot_callback_group = MutuallyExclusiveCallbackGroup()
        self.traj_point_callback_group = MutuallyExclusiveCallbackGroup()
        self.nvblox = os.getenv("NVBLOX", False) == "true"
        self.debug = os.getenv("DEBUG", False) == "true"
        self.do_not_persist_voxels = os.getenv("DO_NOT_PERSIST_VOXELS", False) == "true"
        print(" Running with flags: ")
        print(f"NVBLOX: {self.nvblox}")
        print(f"DEBUG: {self.debug}")
        print(f"DO_NOT_PERSIST_VOXELS: {self.do_not_persist_voxels}")
        # Initialize CuroboMotionPlanner
        self.curoboMotion = CuroboMotionPlanner(cfg)

        if self.nvblox:
            self.curoboMotion.setup_motion_planner_with_collision_avoidance()
        else:
            self.curoboMotion.setup_motion_planner()  # Warmup happens here
            self.curoboMotion.setup_ik_solver()

        # Load the world and robot configurations for ISAAC SIM
        self.world_cfg = self.curoboMotion.world_cfg
        self.robot_cfg = self.curoboMotion.robot_cfg
        self.j_names = self.curoboMotion.j_names
        self.latest_joint_state = None
        self.start_js = None

        # Create the service
        self.target_srv = self.create_service(
            MoveArm, 
            'target_pose', 
            self.target_pose_callback, 
            callback_group=self.srv_callback_group
            )
        
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
        self.torso_command = self.create_publisher(
            Float64MultiArray,
            'torso_joint_angles',
            10
        )
        # Subscriber for manipulator status messages
        self.manipulator_status_subscriber = self.create_subscription(
            Int32,
            "robot_arm_status",
            self.manipulator_status_callback,
            10,
            callback_group=self.arm_callback_group
        )
        self.toros_status_subscriber = self.create_subscription(
            Int32,
            "torso_status",
            self.torso_status_callback,
            10,
            callback_group=self.arm_callback_group
        )

        self.last_traj_sub = self.create_subscription(
            JointTrajectoryPoint,
            "last_traj_point",
            self.last_traj_callback,
            10,
            callback_group=self.traj_point_callback_group
        )

        self.robot_status = None
        self.torso_status = None
        self.torso_joint = ["torso_lift_joint"]

                
        self.lock = threading.Lock()
        self.published_trajectory = None
        self.last_traj_point = None
        self.reversed_traj = None

        self.get_logger().info('Curobo Trajectory Node has been started.')

    def last_traj_callback(self, msg):
        with self.lock:
            self.last_traj_point = msg.positions

    def udpate_env(self, msg):
        self.update_collision_world(persist = msg.data)
        
    def wait_for_idle(self):
        # Wait for the robot to become idle using the condition variable
        while rclpy.ok():
            with self.lock:
                # print(f"Robot status: {self.robot_status}")
                if self.robot_status not in [Status.ACTIVE, Status.PREEMPTING]:
                    break
            time.sleep(0.5)  # Sleep to prevent busy waiting
            self.get_logger().info("Robot is not idle.")

    def wait_for_torso(self):
        # Wait for the robot to become idle using the condition variable
        while rclpy.ok():
            with self.lock:
                # print(f"Robot status: {self.robot_status}")
                if self.torso_status not in [Status.ACTIVE, Status.PREEMPTING]:
                    break
            time.sleep(0.5)  # Sleep to prevent busy waiting
            self.get_logger().info("Robot is not idle.")

    def gripper_status_callback(self, msg):
        if self.gripper_status != msg.data:
            # print(f"Changed gripper status: {msg.data}")
            self.gripper_status = msg.data
    def torso_status_callback(self, msg):
        with self.lock:
            self.torso_status = Status(msg.data)

    def manipulator_status_callback(self, msg):
        # Update the robot status and notify waiting threads
        with self.lock:
            # print(msg.data)
            self.robot_status = Status(msg.data)
        # self.get_logger().info(f"Robot status updated: {self.robot_status}")
        # self.condition.notify_all()  # Notify all waiting threads
        
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
            
            self.torso_js = [joint_positions[joint_name] for joint_name in self.torso_joint]
        except KeyError as e:
            self.curoboMotion.j_names
            self.get_logger().error(f'Joint name {e} not found in joint states.')
            return None
        
        return initial_js
    
    def publish_until_moving(self, joint_trajectory_msg):
        while rclpy.ok():
            with self.lock:
                print(f"Robot status: {self.robot_status}")
                if self.robot_status not in [Status.ACTIVE, Status.PREEMPTING]:
                    self.trajectory_publisher.publish(joint_trajectory_msg)
                    print("Published Trajectory")
                else:
                    break
            time.sleep(0.5)  # Sleep to prevent busy waiting
            self.get_logger().info("Waiting for robot to receive message")

    def target_pose_callback(self, request, response):
        response.success = False
        request_pose = request.target_pose
        request_type = MotionType(int(request.traj_type.data))
        print(f"Request type: {request_type}")
        print(f"Request position: {request_pose.pose.position}")
        print(f"Request orientation: {request_pose.pose.orientation}")
        initial_js = self.get_current_joint_positions() # List of joint positions
        pose = JointStateC.from_position(
                position=self.curoboMotion.tensor_args.to_device(initial_js),
                joint_names=self.j_names[0 : len(initial_js)],
            )
        initial_pose = self.curoboMotion.motion_gen.compute_kinematics(pose)
        print("initial pose: ")
        print(initial_pose.ee_pos_seq[0].cpu().numpy().tolist() + initial_pose.ee_quat_seq[0].cpu().numpy().tolist())
    
        # print(f"Initial Pose: {initial_pose}")
        if request_pose is None or initial_js is None:
            self.get_logger().error('Check the request pose and initial joint positions.')
            response.success = False
            self.published_trajectory = None
            return response
        
        target_pose = self.convert_pose_to_target_format(request_pose) # List: [x, y, z, qw, qx, qy, qz]

        if target_pose is None:
            self.get_logger().error('Failed to find a feasible IK solution. Check Reachability.')
            response.success = False
            self.published_trajectory = None
            return response

        offset = 0.0
        if request_type == MotionType.FREESPACE:
            offset = -0.25 # -Z offset w.r.t ee goal frame
            self.curoboMotion.release_constraint()
            potential_poses = self.curoboMotion.generate_yaw_poses(target_pose)
            print(potential_poses)
            n_succ, target_pose = self.curoboMotion.compute_kinematics_batch(initial_js, potential_poses)
            print("Number of successful poses: ", n_succ)
            # print("Best Pose: ", target_pose)
            
        elif request_type == MotionType.FREESPACE_ARUCO:
            offset = -0.32
            self.curoboMotion.release_constraint()
            potential_poses = self.curoboMotion.generate_yaw_poses(target_pose)
            n_succ, target_pose = self.curoboMotion.compute_kinematics_batch(initial_js, potential_poses)
            print("Number of successful poses: ", n_succ)
            # print("Best Pose: ", target_pose)
        elif request_type == MotionType.BOX_IN:
            self.curoboMotion.scale_velocity(0.5)
            offset = 0.01
            self.curoboMotion.set_constraint()
            # self.curoboMotion.release_constraint()
        elif request_type == MotionType.SHELF_IN:
            self.curoboMotion.scale_velocity(0.6)
            offset = 0.02
            self.curoboMotion.set_constraint()
        elif request_type == MotionType.DROP_IN:
            offset = -0.10
            self.curoboMotion.set_constraint()
        elif request_type == MotionType.BOX_OUT or request_type == MotionType.SHELF_OUT:
            joint_trajectory_msg = self.create_joint_trajectory_message(self.reversed_traj)
            # self.publish_until_moving(joint_trajectory_msg)
            # self.wait_for_idle()
            self.trajectory_publisher.publish(joint_trajectory_msg)
            response.success = True
            return response
        elif request_type == MotionType.RESTOCK_HOME:
            offset = 0.0
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", True)
            target_js = [0.04053, 1.4964325, -3.116786, 1.518171482376709, 0.00017877879488468335, 1.6613113543554687, -0.0004807466445803637]

        else:
            self.get_logger().error('Invalid trajectory type.')
            response.success = False
            self.published_trajectory = None
            return response

        intermediary_pose = self.curoboMotion.compute_intermediary_pose(target_pose, offset) # List: [x, y, z, qw, qx, qy, qz]

        intermediary_ik, goal_ik = self.curoboMotion.ik_goal_generator(initial_js, intermediary_pose, target_pose)
        if intermediary_ik is None or goal_ik is None:
            self.get_logger().error('Failed to find a feasible IK solution. Check Reachability.')
            response.success = False
            self.published_trajectory = None
            return response
        
        print("Start Planning Trajectory")
        if request_type == MotionType.BOX_IN or request_type == MotionType.SHELF_IN or request_type == MotionType.DROP_IN:
            trajectory = self.curoboMotion.generate_trajectory(
                initial_js=initial_js,
                goal_ee_pose=target_pose,
                goal_js_pose=None,
            )
            if trajectory is None:
                self.get_logger().error('Failed to generate trajectory.')
                response.success = False
                self.published_trajectory = None
                return response
            self.reversed_traj = deepcopy(trajectory)
            self.reversed_traj["positions"].reverse()
            self.reversed_traj["velocities"] = [
                [-v for v in velocity] for velocity in reversed(trajectory["velocities"])
            ]

# Reverse the accelerations and multiply by -1
            self.reversed_traj["accelerations"] = [
                [-a for a in acceleration] for acceleration in reversed(trajectory["accelerations"])
            ]

            # if request_type == MotionType.DROP_IN:
            #     self.reversed_traj = trajectory["positions"].reverse()
            # else:
            #     preempt_idx = self.find_position_in_solution(trajectory, self.last_traj_point)
            #     clipped_trajectory = {
            #         "positions": trajectory["positions"][:preempt_idx],
            #         "velocities": trajectory["velocities"][:preempt_idx],
            #         "accelerations": trajectory["accelerations"][:preempt_idx],
            #         "jerks": trajectory["jerks"][:preempt_idx],
            #         "interpolation_dt": trajectory["interpolation_dt"],
            #         "raw_data": trajectory["raw_data"],
            #     }
            #     clipped_trajectory["positions"].reverse()
            #     clipped_trajectory["velocities"].reverse()
            #     clipped_trajectory["accelerations"].reverse()
            #     clipped_trajectory["jerks"].reverse()
                
            #     self.reversed_traj = clipped_trajectory
        elif request_type in [MotionType.FREESPACE, MotionType.FREESPACE_ARUCO]:
            trajectory = self.curoboMotion.generate_trajectory(
                initial_js=initial_js,
                goal_ee_pose=None,
                goal_js_pose=intermediary_ik,
            )
        elif request_type == MotionType.RESTOCK_HOME:
            trajectory = self.curoboMotion.generate_trajectory(
                initial_js=initial_js,
                goal_ee_pose=None,
                goal_js_pose=target_js,
            )
           
        if trajectory is None:
            self.get_logger().error('Failed to generate trajectory.')
            response.success = False
            self.published_trajectory = None
            return response
        
        joint_trajectory_msg = self.create_joint_trajectory_message(trajectory)
        self.published_trajectory = joint_trajectory_msg
        # print(self.published_trajectory)
        # self.publish_until_moving(joint_trajectory_msg)
        self.trajectory_publisher.publish(joint_trajectory_msg)
    
        # self.trajectory_publisher.publish(joint_trajectory_msg)
        self.start_js = initial_js

        # self.get_logger().info('Published joint trajectory.')
        response.success = True
        time.sleep(1)
        self.get_logger().info("Response success: True")

        self.wait_for_idle()

        # if request_type == MotionType.BOX_OUT:
        #     self.curoboMotion.create_and_attach_object(target_pose, initial_js)
        response.success = True        
        print("Response success: True")
        return response
    
    def find_position_in_solution(self, solution_dict, target_position, tolerance=1e-3):
        """
        Finds the index of a target position in the solution dictionary's positions list.

        :param solution_dict: A dictionary containing the trajectory data, including "positions".
        :param target_position: The position array (list or array) to find.
        :param tolerance: Tolerance for floating-point comparison.
        :return: Index of the matching position in the solution_dict, or -1 if not found.
        """
        target_array = np.array(target_position)
        positions = solution_dict["positions"]

        for i, pos in enumerate(positions):
            if np.allclose(pos, target_array, atol=tolerance):
                return i

        return -1  # Return -1 if no match is found
    
    @staticmethod
    def convert_pose_to_target_format(pose_msg):
        # Extract the pose from PoseStamped
        pose = pose_msg.pose

        # Convert ROS2 Pose to the format expected by CuroboMotionPlanner
        # pose_curobo = [pose.position.x, pose.position.y, pose.position.z,
        #                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] 

        # HARD CODING ORIENTATION FOR BOX PICKING BECAUSE ITS A GIVEN --> BASED ON SUCTION_STATUS
        pose.position.x -= 0.086875
        pose.position.z += 0.37743
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

        print(time_from_start)

        return joint_trajectory_msg
    

if __name__ == "__main__":
    rclpy.init()
    cfg = "fetch.yml"
    # cfg = "config/fetch.yml"
    curobo_node = CuroboTrajectoryNode(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(curobo_node)
    
    try:
        # Spin the executor to manage callbacks with multiple threads

        executor.spin()

    except KeyboardInterrupt:
        pass
    finally:
        # Ensure proper shutdown
        executor.shutdown()  # Stop the executor
        curobo_node.destroy_node()
        rclpy.shutdown()






        ########## NAV RETRIEVAL ##########
        # X:-6.29472
        # Y: 5.16153
        # YAW: -13.132619 DEGREES

        ########## END EFFECTOR JS POSE ##########

        # - 0.169388641204834
        # - 0.6278159478271484
        # - -3.0730677736816405
        # - 1.909336738876953
        # - -0.2694181648368835
        # - -1.1546938755944824
        # - -0.22367496302261353
