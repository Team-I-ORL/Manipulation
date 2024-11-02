import carb
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "width": "1920",
    "height": "1080",
})

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
import omni.graph.core as og
import usdrt.Sdf

from helper import add_extensions, add_robot_to_scene, VoxelManager
from curobo.geom.types import WorldConfig, Cuboid, Cylinder
from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


import numpy as np
from typing import List
import rclpy
from rclpy.node import Node
from CuroboMotionPlanner import CuroboMotionPlanner
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
import torch
from matplotlib import cm

add_extensions(simulation_app) # Also enables ROS2
curoboMotion = CuroboMotionPlanner("fetch.yml")

class Subscriber(Node):
    def __init__(self):
        super().__init__("tutorial_subscriber")

        # setting up the world with a cube
        # self.timeline = omni.timeline.get_timeline_interface()

        self.ros_world = World(stage_units_in_meters=1.0)
        stage = self.ros_world.stage
        # self.ros_world.scene.add_default_ground_plane()

        # add and load fetch
        robot_cfg = curoboMotion.robot_cfg
        self.j_names = curoboMotion.j_names
        self.q_start = curoboMotion.q_start

        world_cfg = curoboMotion.world_cfg
        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")

        usd_help = UsdHelper()

        usd_help.load_stage(self.ros_world.stage)
        usd_help.add_world_to_stage(world_cfg, base_frame="/World")

        self.robot, robot_prim_path = add_robot_to_scene(robot_cfg, self.ros_world)
        self.articulation_controller = self.robot.get_articulation_controller()
        print(robot_prim_path)

        self.trajectory_msg = None
        # setup the ROS2 subscriber here
        self.traj_sub = self.create_subscription(JointTrajectory, "joint_trajectory", self.trajectory_callback, 10)
        self.voxel_sub = self.create_subscription(Float32MultiArray, "voxel_array", self.voxel_callback, 10)

        self.ros_world.reset()
        self.executing_traj = False
        self.initial_state = self.q_start

        self.use_debug_draw = True
        render_voxel_size = 0.02

        if self.use_debug_draw:
            self.voxel_viewer = VoxelManager(100, size=render_voxel_size)
    
    def trajectory_callback(self, data):
        # callback function to set the cube position to a new one upon receiving a (empty) ROS2 message
        if self.executing_traj:
            print("Currently executing a trajectory. Ignoring new trajectory.")
            return

        print("Received new trajectory from topic.")

        # Create a FollowJointTrajectoryGoal from the received trajectory
        self.trajectory_msg = data  # The received JointTrajectory messag$
        self.executing_traj = True
        self.initial_state = data.points[0].positions
        self.joint_name_msg = data.joint_names

    def voxel_callback(self, data):
        # convert flattened data back to 3D array
        voxels = torch.tensor(data.data, dtype=torch.float32).view(-1, 3)
        if self.use_debug_draw:
                    self.draw_points(voxels)
        else:
            voxels = voxels.cpu().numpy()
            self.voxel_viewer.update_voxels(voxels[:, :3])

        self.voxel_viewer.update_voxels(voxels[:, :3])


    def draw_points(self, voxels):
        # Third Party

        # Third Party
        from omni.isaac.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        # if draw.get_num_points() > 0:
        draw.clear_points()
        if len(voxels) == 0:
            return

        jet = cm.get_cmap("plasma").reversed()

        cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
        z_val = cpu_pos[:, 0]

        jet_colors = jet(z_val)

        b, _ = cpu_pos.shape
        point_list = []
        colors = []
        for i in range(b):
            # get list of points:
            point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
            colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 0.8)]
        sizes = [20.0 for _ in range(b)]

#     draw.draw_points(point_list, colors, sizes)
    def run_simulation(self):
        # self.timeline.play()
        i = 0
        self.ros_world.reset()
        self.robot._articulation_view.initialize()
        idx_list = [self.robot.get_dof_index(x) for x in self.j_names]

        self.robot.set_joint_positions(self.initial_state, idx_list)
        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
        )
        
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)
            step_index = self.ros_world.current_time_step_index
            if not self.ros_world.is_playing():
                if i%100 == 0:
                    print("########Click to Play########")
                i += 1
                continue
            

            if step_index < 20:
                continue
            if self.executing_traj:
                self.robot.set_joint_positions(self.initial_state, idx_list)
                print("Received trajectory")
                for point in self.trajectory_msg.points:
                    positions = point.positions
                    joint_names_from_message = self.joint_name_msg  # Names in the JointTrajectory message
                    joint_names_robot = self.j_names  # Names of the robot's joints in Isaac Sim

                    # Build idx_list to map the names in the message to Isaac Sim's internal order
                    idx_list = [joint_names_robot.index(name) for name in joint_names_from_message]

                    # Reorder the positions according to the idx_list (Isaac Sim joint order)
                    ordered_positions = [positions[idx] for idx in idx_list]
                    

                    articulation_action = ArticulationAction(joint_positions=ordered_positions)
                    print(articulation_action.joint_positions)

                    self.articulation_controller.apply_action(articulation_action)
                    
                    for _ in range(2):
                        self.ros_world.step(render=True)
                
                self.executing_traj = False




if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()

