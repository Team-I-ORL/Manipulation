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

from collections import deque

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image, CameraInfo
# Import curobo utilities (update these imports as needed)
from curobo.util_file import get_robot_configs_path, get_assets_path
# Import your CuroboMotionPlanner class
from CuroboMotionPlanner import CuroboMotionPlanner
from curobo.types.state import JointState as JointStateC
from fetch_camera import CameraLoader
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
    REVERSE = 13

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
        self.saved_trajectory_stack = deque(maxlen=10)

        if self.nvblox:
            self.collision_env_sub = self.create_subscription(Bool, 'update_collision_env', self.udpate_env, 10, callback_group=MutuallyExclusiveCallbackGroup())

            self.voxel_pub = self.create_publisher(Float32MultiArray, 'voxel_array', 10)
            self.mesh_pub = self.create_publisher(Mesh, 'mesh', 10)
            ## TODO: Replace continous subsription with invoked subscription for lower network load
            self.cam = CameraLoader(self)
            self.create_subscription( Image, '/head_camera/rgb/image_raw', self.cam.rgb_callback, 1, callback_group=MutuallyExclusiveCallbackGroup())
            self.create_subscription( Image, '/head_camera/depth/image_rect_raw', self.cam.depth_callback,1, callback_group=MutuallyExclusiveCallbackGroup() )
            # self.create_subscription( Pose, '/camera/pose', self.pose_callback)
            self.create_subscription( CameraInfo, '/head_camera/rgb/camera_info', self.cam.intrinsics_callback, 1, callback_group=MutuallyExclusiveCallbackGroup())
            self.cam.set_fetch_camera_pose()
                
        self.lock = threading.Lock()
        self.published_trajectory = None
        self.last_traj_point = None

        self.get_logger().info('Curobo Trajectory Node has been started.')

    def last_traj_callback(self, msg):
        with self.lock:
            self.last_traj_point = msg.positions
            # print(msg)
            print(msg.positions)

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


        if request_type == MotionType.TORSO_DOWN:
            torso_pose = Float64MultiArray()
            torso_pose.data = [float(0.054)]
            self.torso_command.publish(torso_pose)
            self.wait_for_torso()
            time.sleep(7)

            response.success = True
            return response
        elif request_type == MotionType.TORSO_UP:
            torso_pose = Float64MultiArray()
            torso_pose.data = [float(0.24)]
            self.torso_command.publish(torso_pose)
            
            self.wait_for_torso()
            time.sleep(7)
            response.success = True
            return response
            
        print(f"Request pose: {request_pose.pose.position}")
        # self.wait_for_idle()
        # self.robot_status = Status.ACTIVE
        #TODO: SLOWER JOINT VELOCITIES
        self.curoboMotion.scale_velocity(1.0) # 50% Speed
        
        initial_js = self.get_current_joint_positions() # List of joint positions
        
        if request_pose is None or initial_js is None:
            self.get_logger().error('Cannot generate trajectory without current joint positions.')
            response.success = False
            self.published_trajectory = None
            return response
        
        target_pose = self.convert_pose_to_target_format(request_pose) # List: [x, y, z, w, x, y, z]
        if target_pose is None:
            self.get_logger().error('Failed to find a feasible IK solution. Check Reachability.')
            response.success = False
            self.published_trajectory = None
            return response

        if request_type == MotionType.FREESPACE:
            offset = -0.20 # -Z offset w.r.t ee goal frame
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", True)
        
        elif request_type == MotionType.FREESPACE_ARUCO: #11
            offset = -0.32 # -Z offset w.r.t ee goal frame
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", True)
        elif request_type == MotionType.DROP_IN: #12
            offset = -0.10
            self.curoboMotion.set_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)
        elif request_type == MotionType.BOX_OUT:
            offset = -0.25
            self.curoboMotion.set_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)
                
        elif request_type == MotionType.SHELF_OUT:
            offset = -0.20
            self.curoboMotion.set_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)
        elif request_type == MotionType.BOX_IN:
            self.curoboMotion.scale_velocity(1.0)
            offset = 0.01
            self.curoboMotion.set_constraint()
            # self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)
        elif request_type == MotionType.SHELF_IN:
            self.curoboMotion.scale_velocity(1.0)
            offset = 0.02
            self.curoboMotion.set_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)
        elif request_type == MotionType.RANDOM:
            offset = 0.0
            self.curoboMotion.release_constraint()

        elif request_type == MotionType.HOME:
            offset = 0.0
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", True)
            target_js = self.curoboMotion.q_start

        elif request_type == MotionType.RESTOCK_HOME:
            offset = 0.0
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", True)
            target_js = [0.04053, 1.4964325, -3.116786, 1.518171482376709, 0.00017877879488468335, 1.6613113543554687, -0.0004807466445803637]
        
        elif request_type == MotionType.REVERSE:
            offset = 0.0
            self.curoboMotion.release_constraint()
            if self.nvblox:
                self.curoboMotion.world_model.enable_obstacle("world", False)

        else:
            self.get_logger().error('Invalid trajectory type.')
            response.success = False
            self.published_trajectory = None
            return response

        print("Start Planning Trajectory")
        if request_type == MotionType.HOME or request_type == MotionType.RESTOCK_HOME or request_type == MotionType.SHELF or request_type == MotionType.BOX:
            if request_type == MotionType.RESTOCK_HOME:
                self.curoboMotion.scale_velocity(1.0)
            # self.curoboMotion.scale_velocity(0.6)

            # target_js = self.curoboMotion.q_start
            # trajectory = self.curoboMotion.go_home(initial_js, target_js)
            if np.allclose(target_js, initial_js, atol=1e-1):
                print("Already at target joint state.")
                response.success = True
                return response
            
            trajectory = self.curoboMotion.generate_trajectory(
                initial_js=initial_js,
                goal_ee_pose=None,
                goal_js_pose=target_js,
            )

        elif request_type == MotionType.REVERSE:
            self.curoboMotion.scale_velocity(1.0)
            try:
                trajectory = self.saved_trajectory_stack.pop()
                # print("reversed trajectory" , len(trajectory["positions"]))
            except IndexError:
                self.get_logger().error('No saved trajectory to reverse.')
                self.published_trajectory = None
                response.success = False
                return response

        else:
            self.curoboMotion.scale_velocity(1.0)
            if request_type == MotionType.BOX_OUT or request_type == MotionType.SHELF_OUT:
                initial_state = JointStateC.from_position(
                    position=self.curoboMotion.tensor_args.to_device(initial_js),
                    joint_names=self.j_names
                )

                initial_pose = self.curoboMotion.motion_gen.compute_kinematics(initial_state)
                print(initial_pose.ee_pos_seq)
                print(initial_pose.ee_quat_seq)

                target_pose = initial_pose.ee_pos_seq.squeeze().cpu().tolist() + initial_pose.ee_quat_seq.squeeze().cpu().tolist() 
                print(target_pose)
                # target_pose.position = initial_pose.ee_pos + target_pose.pos
                # target_pose.quaternion = initial_pose.ee_quat_seq

            target_pose = self.curoboMotion.compute_intermediary_pose(target_pose, offset)

            trajectory = self.curoboMotion.generate_trajectory(
                initial_js=initial_js,
                goal_ee_pose=target_pose,
                goal_js_pose=None,
            )
            time.sleep(1.0)
        
        if trajectory is None:
            self.get_logger().error('Failed to generate trajectory.')
            response.success = False
            self.published_trajectory = None
            return response

        joint_trajectory_msg = self.create_joint_trajectory_message(trajectory)
        self.published_trajectory = joint_trajectory_msg
        # print(self.published_trajectory)
        self.publish_until_moving(joint_trajectory_msg)
    
        # self.trajectory_publisher.publish(joint_trajectory_msg)
        self.start_js = initial_js

        # self.get_logger().info('Published joint trajectory.')
        response.success = True
        time.sleep(1)
        self.get_logger().info("Response success: True")

        self.wait_for_idle()

        # print(self.last_traj_point)
        # idx = self.find_position_in_solution(trajectory, self.last_traj_point)
        # print(idx)
        # print(len(trajectory["positions"]))
        if request_type in [MotionType.FREESPACE, MotionType.FREESPACE_ARUCO, MotionType.DROP_IN, MotionType.BOX_IN, MotionType.SHELF_IN]: 
            # print(trajectory["positions"][-1])
            self.save_trajectory(trajectory)
        # if request_type == MotionType.BOX_OUT:
        #     self.curoboMotion.create_and_attach_object(target_pose, initial_js)
        response.success = True        
        print("Response success: True")
        return response
    
    def find_position_in_solution(self, solution_dict, target_position, tolerance=1e-1):
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
            print(i)
            if np.allclose(pos, target_array, atol=tolerance):
                return i

        return -1  # Return -1 if no match is found
    
    def save_trajectory(self, trajectory):
        ## revcerse trajectory lists and save in stack
        # reversed_trajectory = trajectory.copy()
        print("last_traj_point: ", self.last_traj_point)
        print("trajectory length: ", len(trajectory["positions"]))
        # print("trajectory raw: ", trajectory["positions"])

        if self.last_traj_point is not None:
            idx = self.find_position_in_solution(trajectory, self.last_traj_point)
            print("stopped at traj idx: ", idx)
            # print(len(trajectory["positions"]))
            clipped_trajectory = {
                "positions": trajectory["positions"][:idx],
                "velocities": trajectory["velocities"][:idx],
                "accelerations": trajectory["accelerations"][:idx],
                "interpolation_dt": trajectory["interpolation_dt"]
            }
            trajectory = clipped_trajectory
            self.last_traj_point = None

        trajectory["positions"].reverse()
        trajectory["velocities"] = [
                [-v for v in velocity] for velocity in reversed(trajectory["velocities"])
            ]
        trajectory["accelerations"] = [
                [-a for a in acceleration] for acceleration in reversed(trajectory["accelerations"])
            ]

        self.saved_trajectory_stack.append(trajectory)
        print("Saved Reverse Trajectory")
        print("Current values in stack: ", len(self.saved_trajectory_stack))
        # print("Traj:: ", trajectory["positions"])

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
    
    def update_collision_world(self, persist):
        self.cam.set_fetch_camera_pose()
        self.camera_data = self.cam.get_data()
        # print("Camera Data : ", self.camera_data)
        if self.camera_data is None:
            raise ValueError('Camera data not available.')
        # add in robot planning base frame - should be a constant or can get from tf
        camera_position = self.camera_data['pose']['position'] #[0,0,0]
        camera_orientation = self.camera_data['pose']['orientation'] #[0,0,0,1]
        self.camera_pose = PoseC(
            position=self.curoboMotion.tensor_args.to_device(camera_position),
            quaternion=self.curoboMotion.tensor_args.to_device(camera_orientation), normalize_rotation=True
            )
        voxels, mesh = self.curoboMotion.update_blox_from_camera(self.camera_data, self.camera_pose, persist)
        # print("Mesh :: ", mesh)
        if voxels is not None:
            self.publish_voxel_array(voxels.cpu().numpy())
        if mesh is not None:
            self.publish_mesh_array(mesh)
        
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

    def publish_mesh_array(self, mesh):
        ros_mesh = Mesh()

        # Convert vertices to geometry_msgs/Point format
        for vertex in mesh.vertices:
            point = Point(x=float(vertex[0]), y=float(vertex[1]), z=float(vertex[2]))
            ros_mesh.vertices.append(point)

        # Convert faces to MeshTriangle format
        for face in mesh.faces:
            triangle = MeshTriangle(vertex_indices=[int(face[0]), int(face[1]), int(face[2])])
            ros_mesh.triangles.append(triangle)

        self.mesh_pub.publish(ros_mesh)
        self.get_logger().info(f"Published mesh with {len(ros_mesh.vertices)} vertices and {len(ros_mesh.triangles)} triangles.")

    def debug_loop(self):
        while rclpy.ok():
            print("In Debug Loop")
            target_pose_cuboid = Cuboid("t_pose", dims=[3, 3, 5.0], pose=[0, 0, 0, 1, 0, 0, 0])
            self.update_collision_world(persist=True)
            time.sleep(0.5)

if __name__ == "__main__":
    rclpy.init()
    cfg = "fetch.yml"
    # cfg = "config/fetch.yml"
    curobo_node = CuroboTrajectoryNode(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(curobo_node)
    
    try:
        # Spin the executor to manage callbacks with multiple threads
        if curobo_node.debug:
            curobo_node.debug_thread = threading.Thread(target=curobo_node.debug_loop)
            curobo_node.debug_thread.start()

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
