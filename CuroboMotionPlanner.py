import torch
# CuRobo
# Standard Library
from typing import List, Optional
import cv2

# TODO: POSE CONSTRAINTS FOR DIFFERENT FLAGS (OPERATION PICK OR PLACE)
# Implemented picking/placing pose cost metric based on suction_status (NEED TO TEST)

# 4 MOTION PRIMS (service call with specific type)
# 1. FREESPACE
# 2. LINEAR Z (PULL UP)
# 3. RETREIVE (GO INTO THE SHELF)
# 4. INVERSE RETREIVE (GO OUT OF THE SHELF)
import curobo.geom.transform as Transform
import math
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
    MotionGenResult,
    MotionGenPlanConfig,
    PoseCostMetric
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util.trajectory import InterpolateType, get_spline_interpolated_trajectory

from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.types import ReacherSolveType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.camera import CameraObservation

import yaml
import numpy as np
from math import cos, sin, radians
def print_cuda_memory(tag=None):
    if tag:
        print(f"[{tag}] CUDA Memory Report:")
    else:
        print("CUDA Memory Report:")
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB")
    print("-" * 40)
class CuroboMotionPlanner:
    def __init__(self, cfg):        
        # Load Robot Configuration
        # self.robot_file = load_yaml(join_path(get_robot_configs_path(), cfg))

        self.robot_file = load_yaml(join_path(get_robot_configs_path(), "fetch.yml"))

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
        self.last_added_object = None

        # collision parameters
        self.voxel_size = 0.02

        #debug 
        self.show_window = False

        # IK Checker
        self.ik_solver = None

    def load_stage(self):
        # # Load fetch mobile base from world config path


        # world_cfg = WorldConfig(cuboid=mobile_base_cfg.cuboid)

        world_cfg = WorldConfig.from_dict(
                {
                    "blox": {
                        "world": {
                            "pose": [0, 0, 0, 1, 0, 0, 0],
                            "integrator_type": "occupancy",
                            "voxel_size": 0.02,
                        }
                    }
                }
            )
        ground = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        ground.cuboid[0].name += "_cuboid"
        ground.cuboid[0].pose[2] = -0.1

        world_cfg.add_obstacle(ground.cuboid[0])
        # world_cfg = WorldConfig(cuboid=ground.cuboid)

        # world_cfg = WorldConfig.create_obb_world(world_cfg)
        return world_cfg   
    
    def setup_motion_planner(self):
        # Configure MotionGen
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_cache={"obb": 200, "mesh": 200},
            interpolation_dt=0.1,
            use_cuda_graph=False,
            project_pose_to_goal_frame=True,
            minimize_jerk=True,

            trajopt_tsteps=64,
            num_trajopt_seeds=3,
            num_graph_seeds=1,
            collision_activation_distance=0.001,
            self_collision_check=True,
            maximum_trajectory_dt=4.0,
            fixed_iters_trajopt=None,
            finetune_trajopt_iters=300,
            num_ik_seeds = 3,
            # position_threshold=0.1,
            # rotation_threshold=0.1,
        )

        self.motion_gen = MotionGen(motion_gen_config)

        print("Curobo warming up...")
        self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=True, parallel_finetune=True)
        print("CuRobo is Ready")

        self.freespace_cost = PoseCostMetric(
            hold_partial_pose = False,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([0, 0, 0, 0, 0, 0]),
        )

        self.linear_cost = PoseCostMetric(
            hold_partial_pose = True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([0, 1, 1, 0, 1, 1]),
        )

        config_args = {
            'max_attempts': 10,
            'enable_graph': False,
            'enable_finetune_trajopt': True,
            'partial_ik_opt': False,
            'ik_fail_return': 10,
            # 'parallel_finetune': False,
            'enable_graph_attempt': None,
            # 'pose_cost_metric': pose_cost,
        }
        
        self.plan_config = MotionGenPlanConfig(**config_args)
        self.trajopt_solver = self.motion_gen.trajopt_solver

    def setup_motion_planner_with_collision_avoidance(self):
        print("world_cfg", self.world_cfg)
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_cache={"obb": 200, "mesh": 1000, "voxels":10000},
            trajopt_tsteps=50,
            interpolation_steps=10000,
            collision_checker_type=CollisionCheckerType.BLOX,
            interpolation_dt=0.1,
            use_cuda_graph=True,
            project_pose_to_goal_frame=True,
            minimize_jerk=True,
            # num_trajopt_seeds=5,
            num_trajopt_seeds=4,
            # trajopt_seed_ratio={"linear:": 0.8, "bias": 0.2},
            # use_start_state_as_retract=True,
            # interpolation_type=InterpolateType.CUBIC,
            num_graph_seeds=6,
            collision_activation_distance=0.01,
            self_collision_check=True,
            self_collision_opt = True,
            maximum_trajectory_dt=2.0,
            # fixed_iters_trajopt=None,
            # finetune_trajopt_iters=None,
            num_ik_seeds = 128,
            grad_trajopt_iters=200,
            position_threshold=0.01,
            rotation_threshold = 0.05,
            optimize_dt = True,
            # rotation_threshold=0.1,
        )



        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True, parallel_finetune=True)

        self.world_model = self.motion_gen.world_collision
        print("self.world_model", self.world_model)
    
        self.freespace_cost = PoseCostMetric(
            hold_partial_pose = False,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([0, 0, 0, 0, 0, 0]),
        )

        self.linear_cost = PoseCostMetric(
            hold_partial_pose = True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([0, 1, 1, 0, 1, 1]),
        )

        config_args = {
            'max_attempts': 10,
            'enable_graph': True,
            'enable_finetune_trajopt': True,
            'partial_ik_opt': False,
            'ik_fail_return': 100,
            'use_start_state_as_retract': True,
            # 'parallel_finetune': False,
            # 'enable_graph_attempt': None,
            # 'pose_cost_metric': pose_cost,
        }
        
        self.plan_config = MotionGenPlanConfig(**config_args)
        self.trajopt_solver = self.motion_gen.trajopt_solver

    def setup_ik_solver(self):
        ik_solver_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            num_seeds=20,
            # use_cuda_graph=False,
        )
        self.ik_solver= IKSolver(ik_solver_config)


    def compute_kinematics_batch(self, 
                                joint_state: List[float],
                                goal_poses: List[float]):
        
        joint_state = self.tensor_args.to_device([joint_state])
        goal_poses = self.convert_xyzw_poses_to_curobo(goal_poses)

        n_seeds = len(goal_poses)
        batch = n_seeds
        initial_seed = joint_state.clone().unsqueeze(0).repeat(n_seeds, batch, 1)

        ik_results = self.motion_gen.ik_solver.solve_any(
            solve_type=ReacherSolveType.BATCH,
            retract_config=joint_state,
            goal_pose=goal_poses,
            seed_config=initial_seed,
        )
        
        n_success = np.sum(ik_results.success.cpu().numpy())
        print(f"Successes: {n_success}/{n_seeds}")
        print(ik_results.debug_info)
        min_norm = np.inf
        for ik_result in ik_results.js_solution.position:
            norm = torch.norm(ik_result - joint_state)
            if norm.cpu() < min_norm:
                min_norm = norm
                best_ik = ik_result.cpu().numpy().tolist()
                print(best_ik)
        pose = JointState.from_position(
                position=self.tensor_args.to_device(best_ik),
                joint_names=self.j_names[0 : len(best_ik)],
            )
        best_pose = self.motion_gen.compute_kinematics(pose)
        
        best_pose_out = best_pose.ee_pos_seq[0].cpu().numpy().tolist() + best_pose.ee_quat_seq[0].cpu().numpy().tolist()
        print("Best Pose: ", best_pose_out)
        return n_success, best_pose_out

    def compute_kinematics_single(self, 
                                joint_state: List[float],
                                goal_pose: List[float]):
        
        joint_state = self.tensor_args.to_device([joint_state])
        initial_seed = joint_state.detach().clone().unsqueeze(0)
        goal_pose = Pose(
            position=self.tensor_args.to_device(goal_pose[0:3]),
            quaternion=self.tensor_args.to_device(goal_pose[3:]),
        )

        ik_result = self.ik_solver.solve_single(
                    goal_pose=goal_pose,
                    retract_config=joint_state,
                    seed_config=initial_seed,
                    )
        success = np.sum(ik_result.success.cpu().numpy()) > 0
        ik_solution = ik_result.js_solution.position.cpu().squeeze().numpy().tolist()
        return success, ik_solution
    @staticmethod
    def convert_xyzw_poses_to_curobo(poses: List[List[float]]) -> Pose:
        """Converts xyzw poses to CuRobo's Pose format."""
        poses = np.array(poses)
        pos = torch.from_numpy(poses[:,:3].copy())
        quat = torch.from_numpy(poses[:,[6,3,4,5]].copy())

        pose = Pose(
            position=pos.float().cuda(),
            quaternion=quat.float().cuda(),
        )
        return pose
    def generate_yaw_poses(self, goal_ee_pose: List[float]) -> List[Pose]:
        """
        Generates poses with different yaw angles (0, 90, 180, 270 degrees) in the gripper frame.

        :param goal_ee_pose: A list containing the [x, y, z, qx, qy, qz, qw] of the end-effector pose in the base_link frame.
        :return: List of poses with different yaw rotations as lists.
        """
        position = goal_ee_pose[0]
        quaternion = goal_ee_pose[1]
        print(position)
        print(quaternion)
        # Convert quaternion to a rotation matrix
        def quaternion_to_matrix(quaternion):
            qw, qx, qy, qz = quaternion
            return [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]

        # Convert rotation matrix back to a quaternion
        def matrix_to_quaternion(matrix):
            m = matrix
            t = m[0][0] + m[1][1] + m[2][2]
            if t > 0:
                s = 0.5 / math.sqrt(t + 1.0)
                qw = 0.25 / s
                qx = (m[2][1] - m[1][2]) * s
                qy = (m[0][2] - m[2][0]) * s
                qz = (m[1][0] - m[0][1]) * s
            elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
                s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
                qw = (m[2][1] - m[1][2]) / s
                qx = 0.25 * s
                qy = (m[0][1] + m[1][0]) / s
                qz = (m[0][2] + m[2][0]) / s
            elif m[1][1] > m[2][2]:
                s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
                qw = (m[0][2] - m[2][0]) / s
                qx = (m[0][1] + m[1][0]) / s
                qy = 0.25 * s
                qz = (m[1][2] + m[2][1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
                qw = (m[1][0] - m[0][1]) / s
                qx = (m[0][2] + m[2][0]) / s
                qy = (m[1][2] + m[2][1]) / s
                qz = 0.25 * s
            return [qx, qy, qz, qw]

        rotation_matrix = quaternion_to_matrix(quaternion)

        # Define yaw rotations in the gripper frame
        yaw_angles = [0, 90, 180, 270]
        poses = []

        for angle in yaw_angles:
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            yaw_rotation_matrix = [
                [1.0,   0.0,     0.0],
                [0.0,   cos_a,  -sin_a],
                [0.0,   sin_a,   cos_a]
            ]

            # Apply yaw rotation
            new_rotation_matrix = np.matmul(rotation_matrix, yaw_rotation_matrix)
            new_quaternion = matrix_to_quaternion(new_rotation_matrix)

            # Combine position and orientation
            poses.append(position + new_quaternion)

        return poses

    def compute_intermediary_pose(self, goal_ee_pose: List[float], offset_z: float):
        """
        Computes an intermediary pose by offsetting the given goal pose along the z-axis of the end-effector.
        
        :param goal_ee_pose: A list containing the [x, y, z, qx, qy, qz, qw] of the end-effector pose in the base_link frame.
        :param offset_z: The offset distance along the z-axis of the end-effector frame (positive values move along the gripper's z-axis).
        
        :return: Pose object representing the intermediary pose.
        """
        # Convert the goal position and quaternion to tensors
        position = self.tensor_args.to_device(goal_ee_pose[0:3])
        
        # Convert quaternion to tensor and add an extra dimension for batch size
        quaternion = self.tensor_args.to_device(goal_ee_pose[3:]).unsqueeze(0)

        # Convert the quaternion to a 3x3 rotation matrix
        rotation_matrix = Transform.quaternion_to_matrix(quaternion)

        # Define the translation offset in the local z-axis of the end-effector
        local_offset = self.tensor_args.to_device([offset_z, 0.0, 0.0])

        # Apply the translation by transforming the local offset with the rotation matrix
        # This gives the offset in the world (base_link) frame
        offset_in_world_frame = torch.matmul(rotation_matrix, local_offset)

        # Compute the intermediary position by adding the offset to the original goal position
        intermediary_position = position + offset_in_world_frame

        # The orientation (quaternion) remains the same as the goal pose
        intermediary_orientation = quaternion

        # Create the new intermediary pose
        # intermediary_pose = intermediary_position.cpu().numpy().tolist() + quaternion.cpu().numpy().tolist()
        
        intermediary_pose = Pose(
            position=intermediary_position,
            quaternion=intermediary_orientation
        )
        # print(intermediary_pose)
        return intermediary_pose



    def compute_intermediary_poses(self, goal_ee_pose: List[float], offset_z: float):
        """
        Computes an intermediary pose by offsetting the given goal pose along the z-axis of the end-effector.
        
        :param goal_ee_pose: A list containing the [x, y, z, qx, qy, qz, qw] of the end-effector pose in the base_link frame.
        :param offset_z: The offset distance along the z-axis of the end-effector frame (positive values move along the gripper's z-axis).
        
        :return: Pose object representing the intermediary pose.
        """
        # Convert the goal position and quaternion to tensors
        position = self.tensor_args.to_device(goal_ee_pose[0:3])
        
        # Convert quaternion to tensor and add an extra dimension for batch size
        quaternion = self.tensor_args.to_device(goal_ee_pose[3:]).unsqueeze(0)

        # Convert the quaternion to a 3x3 rotation matrix
        rotation_matrix = Transform.quaternion_to_matrix(quaternion)

        # Define the translation offset in the local z-axis of the end-effector
        local_offset = self.tensor_args.to_device([offset_z, 0.0, 0.0])

        # Apply the translation by transforming the local offset with the rotation matrix
        # This gives the offset in the world (base_link) frame
        offset_in_world_frame = torch.matmul(rotation_matrix, local_offset)

        # Compute the intermediary position by adding the offset to the original goal position
        intermediary_position = position + offset_in_world_frame

        # The orientation (quaternion) remains the same as the goal pose
        intermediary_orientation = quaternion

        # Create the new intermediary pose
        intermediary_pose = intermediary_position.cpu().numpy().tolist() + quaternion.cpu().numpy().tolist()
        
        # print(intermediary_pose)
        return intermediary_pose
    def set_constraint(self):
        self.plan_config.pose_cost_metric = self.linear_cost
        
    def release_constraint(self):
        self.plan_config.pose_cost_metric = self.freespace_cost
    def scale_velocity(self, scale):
        self.plan_config.time_dilation_factor = scale
        print("Setting Time Dilation Factor (Speed): ", self.plan_config.time_dilation_factor)


    # # def create_and_attach_object(self, ee_pos, initial_js) -> None:
    # #     cu_js = JointState.from_position(
    # #         position=self.tensor_args.to_device(initial_js),
    # #         joint_names=self.j_names[0 : len(initial_js)],
    # #     )

    # #     # TODO: This function does not clear out previously attached objects and will cause memory leakage!
    # #     # Can be avoided for now simply by calling this function sparingly
    # #     if self.last_added_object is not None:
    # #         self.world_cfg.remove_obstacle(self.last_added_object)
    # #     # USE FK TO FIND WHERE TO PLACE OBJECT
    # #     ee_pos = self.motion_gen.ik_solver.fk(cu_js.position).ee_position.squeeze().cpu().numpy()
    # #     print("EE Position: ", ee_pos)
    # #     object = Cuboid(
    # #         name="object",
    # #         pose=[ee_pos[0], ee_pos[1], ee_pos[2], 1.0, 0.0, 0.0, 0.0],
    # #         dims=[0.1, 0.2, 0.055], # length, width, height (0.1, 0.2 are like shelf orientation)
    # #         )
        
        
    # #     self.world_cfg.add_obstacle(object)
    # #     self.last_added_object = 'object'
    # #     self.motion_gen.update_world(self.world_cfg)
    # #     self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 20}

    # #     self.motion_gen.detach_object_from_robot()
    # #     self.motion_gen.attach_external_objects_to_robot(
    # #         joint_state=cu_js,
    # #         external_objects=[object],
    # #         surface_sphere_radius=0.005,
    # #         sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
    # #     )
        
    # def detach_obj(self) -> None:
    #     self.detach = True
    #     self.motion_gen.detach_spheres_from_robot()

    def generate_batch_trajectory(self,
                          initial_js: List[float],
                            goal_ee_poses: List[List[float]],
    ): 

        n = len(goal_ee_poses)
        print(n)
        print_cuda_memory("Before Planning")
        vel = [0.0] * len(initial_js)
        accel = [0.0] * len(initial_js)

        start_state = JointState(
            self.tensor_args.to_device([initial_js]*n),
            self.tensor_args.to_device([vel]*n),
            self.tensor_args.to_device([accel]*n),
            tensor_args=self.tensor_args,
        )

        goal_poses = self.convert_xyzw_poses_to_curobo(goal_ee_poses)
        print_cuda_memory("Before Planning")

        print(start_state)
        print(goal_poses)
        motion_gen_result = self.motion_gen.plan_batch(
            start_state, goal_poses, self.plan_config
        )
        # interpolated_solution = motion_gen_result.get_interpolated_plan()
        
        success = motion_gen_result.success.cpu().numpy()
        print(success)
        print("Successes: {} out of {}".format(np.sum(success), len(success)))
        if np.any(success):
            dts = motion_gen_result.optimized_dt.cpu().numpy().tolist()
            
            trajs = motion_gen_result.optimized_plan #MotionGenReulst.optimized_plan

            print(dts)
        else:
            return None
        min_dt = min(dts)  # Find the minimum value
        best_traj_idx = dts.index(min_dt)
        best_traj_pos = trajs.position[best_traj_idx]
        best_traj_vel = trajs.velocity[best_traj_idx]
        best_traj_acc = trajs.acceleration[best_traj_idx]
        opt_plan = JointState(
            position=best_traj_pos,
            velocity=best_traj_vel,
            acceleration=best_traj_acc,
            joint_names=self.j_names[0 : len(initial_js)],
        )
        scaled_dt = min_dt * (1.0 / 0.5) #0.5 IS THE TIME DILATION FACTOR IN TENSOR
        
        opt_plan = opt_plan.scale_by_dt(
            self.tensor_args.to_device(min_dt), 
            self.tensor_args.to_device(scaled_dt)
        )
        solution_dict = {
            "joint_names": trajs.joint_names,
            "positions": opt_plan.position.cpu().squeeze().numpy().tolist(),
            "accelerations": opt_plan.acceleration.cpu().squeeze().numpy().tolist(),
            "velocities": opt_plan.velocity.cpu().squeeze().numpy().tolist(),
            "interpolation_dt": scaled_dt,
            # "raw_data": trajs,
        }

        return solution_dict
    

    def generate_trajectory(self,
                       initial_js: List[float],
                       goal_ee_pose: List[float] = None,
                       goal_js_pose: List[float] = None,
    ): 
        
        # if goal_ee_pose is not None and goal_js is None:
        if goal_ee_pose is not None:

            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )
            
            # goal_pose = Pose(
            #     position=self.tensor_args.to_device(goal_ee_pose[0:3]),
            #     quaternion=self.tensor_args.to_device(goal_ee_pose[3:]),
            # )
            goal_pose = goal_ee_pose
            
            try:
                motion_gen_result = self.motion_gen.plan_single(
                    initial_js, goal_pose, self.plan_config
                )
                print(motion_gen_result.status)
                reach_succ = motion_gen_result.success.item()
                print("Success (should only have one): ", reach_succ)
                interpolated_solution = motion_gen_result.get_interpolated_plan() 

            except Exception as e:
                print("Error in planning trajectory: ", e)
                return None
        elif goal_js_pose is not None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )        
            # goal_js_pose = [1.3056849, 1.4040100, -0.34258141, 1.743283, 0.017052, 1.627947, -0.129718]
            # goal_js_pose = self.q_start
            goal_js = JointState.from_position(
                position=self.tensor_args.to_device([goal_js_pose]),
                joint_names=self.j_names[0 : len(goal_js_pose)],
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single_js(
                    initial_js, goal_js, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()
                print("Success (should only have one): ", reach_succ)
                interpolated_solution = motion_gen_result.get_interpolated_plan() 

            except Exception as e:
                
                print("Error in planning trajectory: ", e)
                return None
        else:
            raise ValueError("Check Goal EE Pose")
        
        
        if reach_succ:      
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
            print("Failed to reach goal")
            return None
        
    # update blox given camera data in dict format and camera pose relative to the planning base frame
    def update_blox_from_camera(self, camera_data, camera_pose, persist=True) -> None:
        if not persist:
            self.world_model.decay_layer("world")
            self.world_model.clear_blox_layer("world")
            print(" Cleared the world layer")

        if persist:
            data_camera = CameraObservation(rgb_image = camera_data["rgba"], depth_image=camera_data["depth"], intrinsics=camera_data["intrinsics"], 
                                            pose=camera_pose)
            print(" camera observation set ")
            self.world_model.add_camera_frame(data_camera, "world")
            self.world_model.process_camera_frames("world", process_aux=True)
            torch.cuda.synchronize()
            self.world_model.update_blox_hashes()

            print("update blox steps done")
            # bounding = Cuboid("t", dims=[2, 2, 2], pose=[0.5, 0.5, 0.5, 1, 0, 0, 0])

        try:
            bounding = Cuboid("t", dims=[3, 3, 3.0], pose=[0, 0, 0, 1, 0, 0, 0])
            voxels = self.world_model.get_voxels_in_bounding_box(bounding, self.voxel_size)
            # mesh = self.world_model.get_mesh_in_bounding_box(bounding, )
            # mesh = self.world_model.get_mesh_from_blox_layer(layer_name = "world", mode="voxel")
            print("Voxels before filtering:", voxels)
        except Exception as e:
            print("Error : ", e)
            voxels = None
        mesh = None
        
        if voxels is not None:
            if voxels.shape[0] > 0:
                voxels = voxels[voxels[:, 2] >= self.voxel_size]
                voxels = voxels[voxels[:, 0] > 0.0]
                # if args.use_debug_draw:
                #     draw_points(voxels)
                # else:
                #     voxels = voxels.cpu().numpy()
                #     voxel_viewer.update_voxels(voxels[:, :3])
                # voxel_viewer.update_voxels(voxels[:, :3])
            else:
                pass
                # if not args.use_debug_draw:
                #     voxel_viewer.clear()
            
            if self.show_window:
                depth_image = camera_data["depth"].cpu().numpy()
                # color_image = camera_data["raw_rgb"].cpu().numpy()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_VIRIDIS
                )
                # color_image = cv2.flip(color_image, 1)
                depth_colormap = cv2.flip(depth_colormap, 1)

                # images = np.hstack((color_image, depth_colormap))

                cv2.namedWindow("NVBLOX", cv2.WINDOW_NORMAL)
                cv2.imshow("NVBLOX", depth_colormap)
                # cv2.imshow("RGB", color_image)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord("q") or key == 27:
                    cv2.destroyAllWindows()

        return voxels, mesh
    
    def clear_bounding_box(
    self,
    cuboid: Cuboid,
    layer_name: Optional[str] = None,
    ):
        """Clear occupied regions of a voxel layer using the given bounding box.

        Args:
            cuboid: Bounding box to clear, defined as a Cuboid object.
            layer_name: Name of the voxel layer where the region will be cleared.
        """
        if layer_name is None:
            print("Layer name must be provided to clear the bounding box.")
            return

        if self.world_model._voxel_tensor_list is None:
            print("Voxel tensor list is not initialized.")
            return

        print("Voxel tensor list:", self.world_model._voxel_tensor_list)
        try:
            # Get the environment index and the voxel grid index
            env_idx = 0  # Assuming a single environment setup
            obs_idx = self.world_model.get_voxel_idx(layer_name, env_idx)

            # Extract the pose and dimensions of the bounding box
            pose_mat = cuboid.pose.get_matrix().view(4, 4).cpu().numpy()
            dims = np.array(cuboid.dims)

            # Calculate the range of voxel indices to clear based on the voxel size
            voxel_size = self.world_model._voxel_tensor_list[0][env_idx, obs_idx, 3].item()
            min_coords = np.floor((pose_mat[:3, 3] - dims / 2) / voxel_size).astype(int)
            max_coords = np.ceil((pose_mat[:3, 3] + dims / 2) / voxel_size).astype(int)

            # Clip the ranges to stay within the bounds of the voxel grid
            voxel_grid = self.world_model._voxel_tensor_list[3][env_idx, obs_idx]
            grid_shape = voxel_grid.shape[:3]
            min_coords = np.clip(min_coords, 0, np.array(grid_shape) - 1)
            max_coords = np.clip(max_coords, 0, np.array(grid_shape) - 1)

            # Iterate over the voxel grid and clear voxels within the bounding box
            for x in range(min_coords[0], max_coords[0] + 1):
                for y in range(min_coords[1], max_coords[1] + 1):
                    for z in range(min_coords[2], max_coords[2] + 1):
                        voxel_grid[x, y, z] = -1.0 * self.max_esdf_distance

            # Update the voxel tensor with the cleared values
            self._voxel_tensor_list[3][env_idx, obs_idx] = voxel_grid
            print(f"Cleared bounding box in layer '{layer_name}' successfully.")
        except ValueError:
            print(f"Layer name '{layer_name}' not found in voxel names.")
        except Exception as e:
            print(f"An error occurred while clearing the bounding box: {e}")







# ##### working params ###### Nov17
#  self.robot_cfg,
#             self.world_cfg,
#             self.tensor_args,
#             collision_cache={"obb": 200, "mesh": 1000, "voxels":10000},
#             trajopt_tsteps=40,
#             interpolation_steps=10000,
#             collision_checker_type=CollisionCheckerType.BLOX,
#             interpolation_dt=0.1,
#             use_cuda_graph=True,
#             project_pose_to_goal_frame=True,
#             minimize_jerk=True,
#             # num_trajopt_seeds=5,
#             num_trajopt_seeds=4,
#             # use_start_state_as_retract=True,
#             # interpolation_type=InterpolateType.CUBIC,
#             num_graph_seeds=4,
#             collision_activation_distance=0.03,
#             self_collision_check=True,
#             maximum_trajectory_dt=2.0,
#             # fixed_iters_trajopt=None,
#             # finetune_trajopt_iters=None,
#             num_ik_seeds = 32,
#             grad_trajopt_iters=200,
#             position_threshold=0.01,
#             optimize_dt = True
#             # rotation_threshold=0.1,

# #################################
############### updated ##########
# self.robot_cfg,
#             self.world_cfg,
#             self.tensor_args,
#             collision_cache={"obb": 200, "mesh": 1000, "voxels":10000},
#             trajopt_tsteps=50,
#             interpolation_steps=5000,
#             collision_checker_type=CollisionCheckerType.BLOX,
#             interpolation_dt=0.08,
#             use_cuda_graph=True,
#             project_pose_to_goal_frame=True,
#             minimize_jerk=True,
#             # num_trajopt_seeds=5,
#             num_trajopt_seeds=4,
#             # trajopt_seed_ratio={"linear:": 0.8, "bias": 0.2},
#             # use_start_state_as_retract=True,
#             # interpolation_type=InterpolateType.CUBIC,
#             num_graph_seeds=6,
#             collision_activation_distance=0.03,
#             self_collision_check=True,
#             self_collision_opt = True,
#             maximum_trajectory_dt=2.0,
#             # fixed_iters_trajopt=None,
#             # finetune_trajopt_iters=None,
#             num_ik_seeds = 128,
#             grad_trajopt_iters=200,
#             position_threshold=0.01,
#             rotation_threshold = 0.1,
#             optimize_dt = True,
#             # rotation_threshold=0.1,
########################################