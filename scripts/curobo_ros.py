import argparse
import os
import rclpy
from rclpy.node import Node
import numpy as np
import threading
# Import ROS2 message types
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
    def __init__(self, cfg):
        super().__init__('curobo_trajectory_node')

        # Initialize CuroboMotionPlanner
        self.curoboMotion = CuroboMotionPlanner(cfg)
        self.curoboMotion.setup_motion_planner()  # Warmup happens here
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
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10)
        self.lock = threading.Lock()
        self.published_trajectory = None

        self.get_logger().info('Curobo Trajectory Node has been started.')

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
        except KeyError as e:
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

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="fetch.yml", 
        help="Configuration file to load"
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode",
        default=False
    )
    parsed_args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init(args=args)

    # Create the node
    curobo_node = CuroboTrajectoryNode(parsed_args.cfg)
    if not parsed_args.sim:
        # Run the node
        rclpy.spin(curobo_node)
    else:
        curobo_node.get_logger().info('Running in simulation mode.')
        spin_thread = threading.Thread(target=rclpy.spin, args=(curobo_node,), daemon=True)
        spin_thread.start()

        import torch
        a = torch.zeros(4, device="cuda:0")

        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

        from omni.isaac.core import World
        from omni.isaac.core.utils.types import ArticulationAction
        from helper import add_extensions, add_robot_to_scene
        from curobo.util.usd_helper import UsdHelper
        import numpy as np
        from typing import List
        import carb


        sim_world = World(stage_units_in_meters=1.0)
        stage = sim_world.stage
        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")

        add_extensions(simulation_app)

        usd_help = UsdHelper()

        usd_help.load_stage(sim_world.stage)

        usd_help.add_world_to_stage(curobo_node.world_cfg, base_frame="/World")
        robot, robot_prim_path = add_robot_to_scene(curobo_node.robot_cfg, sim_world)
        articulation_controller = robot.get_articulation_controller()

        i = 0

        while simulation_app.is_running():
            with curobo_node.lock:
                solutions = curobo_node.published_trajectory
                q_start = curobo_node.start_js
                
            if solutions:
                print(solutions)
                curobo_node.get_logger().info(f"Executing Trajectory {i}")
                if not sim_world.is_playing():
                    sim_world.play()
                step_index = sim_world.current_time_step_index
                if step_index < 2:
                    sim_world.reset()
                    robot._articulation_view.initialize()
                    idx_list = [robot.get_dof_index(x) for x in curobo_node.j_names]
                    robot.set_joint_positions(q_start, idx_list)

                
                if step_index < 20:
                    continue

                for point in solutions.points:
                    positions = point.positions
                    sim_js = robot.get_joints_state()
                    sim_js_names = robot.dof_names
                    
                    if np.any(np.isnan(sim_js.positions)):
                        carb.log_warn("Isaac Sim has returned NAN joint position values.")
                    tensor_args = curobo_node.curoboMotion.tensor_args

                    cu_js = JointStateC(
                        position=tensor_args.to_device(sim_js.positions),
                        velocity=tensor_args.to_device(sim_js.velocities),
                        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                        joint_names=sim_js_names,
                    )
                    cu_js = cu_js.get_ordered_joint_state(curobo_node.curoboMotion.kinematics.joint_names)
                    articulation_action = ArticulationAction(joint_positions=point)
                    articulation_controller.apply_action(articulation_action)

                    for _ in range(10):
                        sim_world.step(render=True)
                        # After executing the trajectory, reset the published trajectory
                with curobo_node.lock:
                    curobo_node.published_trajectory = None
                curobo_node.get_logger().info("Trajectory execution completed.")
            else:
                curobo_node.get_logger().info("No Trajectory to execute")
            
            simulation_app.update()

    # Clean up
    curobo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()