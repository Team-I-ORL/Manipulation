import torch
# CuRobo
# Standard Library
from typing import List, Optional

# TODO: POSE CONSTRAINTS FOR DIFFERENT FLAGS (OPERATION PICK OR PLACE)
# ROS FLAG OF SUCTION GRIPPER STATUS (TRUE FOR SUCKED)
# ROS FLAG FOR TYPE OF OPERATION.

from curobo.geom.types import WorldConfig, Cuboid, Cylinder
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util.trajectory import get_spline_interpolated_trajectory
from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.types import ReacherSolveType

import yaml
import numpy as np
from math import cos, sin, radians

class CuroboMotionPlanner:
    def __init__(self, cfg):        
        # Load Robot Configuration
        self.robot_file = load_yaml(join_path(get_robot_configs_path(), cfg))
        if self.robot_file is None:
            self.robot_file = load_yaml(join_path(get_robot_configs_path(), "fetch.yml"))

        self.robot_cfg = self.robot_file["robot_cfg"]
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.q_start = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

        # Fetch Mobile Base as collision environment
        self.world_cfg = self.load_stage()

        self.tensor_args = TensorDeviceType()
        
        # Motion Planning Config
        self.motion_gen = None
        self.plan_config = None
        self.trajectories = []
    
    def load_stage(self):
        # Load fetch mobile base from world config path
        mobile_base_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_fetch_base.yml"))
        )

        world_cfg = WorldConfig(cylinder=mobile_base_cfg.cylinder)
        world_cfg = WorldConfig.create_obb_world(world_cfg)
        return world_cfg   
    
    def setup_motion_planner(self):
        # Configure MotionGen
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_cache={"obb": 200, "mesh": 100},
            interpolation_dt=0.01,
            trajopt_tsteps=24,
            use_cuda_graph=False,
            project_pose_to_goal_frame=False,
        )

        self.motion_gen = MotionGen(motion_gen_config)

        print("Curobo warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True, parallel_finetune=True)
        print("CuRobo is Ready")

        # MAY NEED BELOW        
        pose_cost = PoseCostMetric(
            hold_partial_pose = True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 0, 0, 0, 0]),
        ) # [rx, ry, rz, tx, ty, tz]
        
        config_args = {
            'max_attempts': 100,
            'enable_graph': False,
            'enable_finetune_trajopt': True,
            'partial_ik_opt': False,
            'parallel_finetune': True,
            'enable_graph_attempt': None,
            # 'pose_cost_metric': pose_cost,
        }
        
        self.plan_config = MotionGenPlanConfig(**config_args)
        self.trajopt_solver = self.motion_gen.trajopt_solver
    
    def generate_trajectory(self,
                       initial_js: List[float],
                       goal_ee_pose: Optional[List[float]] = None,
                       goal_js: Optional[List[float]] = None,
    ):
        if goal_ee_pose is not None and goal_js is None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )
            # initial_ee_pos = self.motion_gen.compute_kinematics(initial_js).ee_pos_seq[0].cpu().numpy()
            
            goal_pose = Pose(
                position=self.tensor_args.to_device(goal_ee_pose[0:3]),
                quaternion=self.tensor_args.to_device(goal_ee_pose[3:]),
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single(
                    initial_js, goal_pose, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()

            except Exception as e:
                print("Error in planning trajectory: ", e)
                return None
            
        elif goal_js is not None and goal_ee_pose is None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )        
            
            goal_js = JointState.from_position(
                position=self.tensor_args.to_device([goal_js]),
                joint_names=self.j_names[0 : len(goal_js)],
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single_js(
                    initial_js, goal_js, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()

            except:
                return None
        else:
            raise ValueError("Either goal_js or goal_ee_pose must be provided.")
        
        if reach_succ:
            interpolated_solution = motion_gen_result.get_interpolated_plan() 
        
            solution_dict = {
                "success": motion_gen_result.success.item(),
                "joint_names": interpolated_solution.joint_names,
                "positions": interpolated_solution.position.cpu().squeeze().numpy().tolist(),
                "velocities": interpolated_solution.velocity.cpu().squeeze().numpy().tolist(),
                "accelerations": interpolated_solution.acceleration.cpu().squeeze().numpy().tolist(),
                "jerks": interpolated_solution.jerk.cpu().squeeze().numpy().tolist(),
                "interpolation_dt": motion_gen_result.interpolation_dt,
                "raw_data": interpolated_solution,
            }
            
            return solution_dict
        
        else:
            return None
        
        # MAY NEED BELOW IN THE FUTURE
    # def create_and_attach_object(self, ee_pos, cu_js, radius: int) -> None:

    #     # TODO: This function does not clear out previously attached objects and will cause memory leakage!
    #     # Can be avoided for now simply by calling this function sparingly

    #     radius *= 0.01
    #     height = 0.10
        
    #     hor_offset = radius # Empirically found
    #     ver_offset = 0.07

    #     dish_cylinder = Cylinder(
    #         name="dish",
    #         pose=[ee_pos[0], ee_pos[1] - hor_offset, ee_pos[2]-ver_offset, 1.0, 0.0, 0.0, 0.0],
    #         radius=radius,
    #         height=height, # same height for all the bowls
    #         )
    #     self.world_cfg.add_obstacle(dish_cylinder)
    #     self.motion_gen.update_world(self.world_cfg)
    #     self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 20}

    #     self.motion_gen.attach_external_objects_to_robot(
    #         joint_state=cu_js,
    #         external_objects=[dish_cylinder],
    #         surface_sphere_radius=0.005,
    #         sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
    #     )
        
    # def detach_obj(self) -> None:
    #     self.detach = True
    #     self.motion_gen.detach_spheres_from_robot()
