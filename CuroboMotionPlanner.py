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
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.camera import CameraObservation

import yaml
import numpy as np
from math import cos, sin, radians

class CuroboMotionPlanner:
    def __init__(self, cfg):        
        # Load Robot Configuration
        # self.robot_file = load_yaml(join_path(get_robot_configs_path(), cfg))

        self.robot_file = load_yaml(cfg)

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
        self.voxel_size = 0.05

        #debug 
        self.show_window = False

        # IK Checker
        self.ik_solver = None

    def load_stage(self):
        # # Load fetch mobile base from world config path
        ground = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        ground.cuboid[0].name += "_cuboid"
        ground.cuboid[0].pose[2] = -10.5

        # world_cfg = WorldConfig(cuboid=mobile_base_cfg.cuboid)

        world_cfg = WorldConfig.from_dict(
                {
                    "blox": {
                        "world": {
                            "pose": [0, 0, 0, 1, 0, 0, 0],
                            "integrator_type": "occupancy",
                            "voxel_size": 0.05,
                        }
                    }
                }
            )
        # world_cfg.add_obb(cuboid=mobile_base_cfg.cuboid)
        world_cfg = WorldConfig(cuboid=ground.cuboid)

        # world_cfg = WorldConfig.create_obb_world(world_cfg)
        return world_cfg   
    
    def setup_motion_planner(self):
        # Configure MotionGen
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_cache={"obb": 500, "mesh": 500},
            interpolation_dt=0.01,
            trajopt_tsteps=32,
            js_trajopt_tsteps=64,
            use_cuda_graph=False,
            project_pose_to_goal_frame=True,
            minimize_jerk=False,
            position_threshold=0.1,
            rotation_threshold=0.1,
        )

        self.motion_gen = MotionGen(motion_gen_config)

        print("Curobo warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True, parallel_finetune=True)
        print("CuRobo is Ready")

        self.freespace_cost = PoseCostMetric(
            hold_partial_pose = False,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([0, 0, 0, 0, 0, 0]),
        )

        self.linear_cost = PoseCostMetric(
            hold_partial_pose = True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 0, 1, 1]),
        )

        config_args = {
            'max_attempts': 100,
            'enable_graph': False,
            'enable_finetune_trajopt': True,
            'partial_ik_opt': True,
            'parallel_finetune': True,
            'enable_graph_attempt': 100,
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
            collision_cache={"obb": 200, "mesh": 1000, "voxels":1000},
            trajopt_tsteps=64,
            collision_checker_type=CollisionCheckerType.BLOX,
            interpolation_dt=0.01,
            use_cuda_graph=False,
            project_pose_to_goal_frame=True,
            minimize_jerk=True,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            collision_activation_distance=0.001,
            self_collision_check=True,
            maximum_trajectory_dt=2.0,
            fixed_iters_trajopt=None,
            finetune_trajopt_iters=300,
            num_ik_seeds = 12,
            position_threshold=0.1,
            rotation_threshold=0.1,
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
            hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 0, 1, 1]),
        )

        config_args = {
            'max_attempts': 10,
            'enable_graph': False,
            'enable_finetune_trajopt': False,
            'partial_ik_opt': False,
            'ik_fail_return': 100,
            'parallel_finetune': False,
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
        :return: List of Pose objects with different yaw rotations.
        """
        # Convert the goal position and quaternion to tensors
        position = goal_ee_pose[0:3]
        quaternion = self.tensor_args.to_device(goal_ee_pose[3:]).unsqueeze(0)

        # Convert the quaternion to a 3x3 rotation matrix
        rotation_matrix = Transform.quaternion_to_matrix(quaternion)
        
        # Define yaw rotations in the gripper frame
        yaw_angles = [0, 90, 180, 270]
        yaw_rotations = []
        
        for angle in yaw_angles:
            # Rotation matrix for yaw in gripper frame (around z-axis of gripper)
            angle_rad = torch.tensor(angle * 3.14159 / 180.0)  # Convert to radians
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            yaw_rotation_matrix = torch.tensor([
                [1.0,  0.0,    0.0],
                [0.0,  cos_a, -sin_a],
                [0.0,  sin_a,  cos_a]
            ], device=self.tensor_args.device)

            # Apply yaw rotation in the gripper frame to the original orientation
            new_rotation_matrix = torch.matmul(rotation_matrix, yaw_rotation_matrix)
            new_quaternion = Transform.matrix_to_quaternion(new_rotation_matrix)
            # print("New Quaternion: ", new_quaternion)
            new_quaternion = new_quaternion.squeeze().cpu().tolist()
            yaw_rotations.append(new_quaternion)

        # Generate poses with different yaw orientations
        poses = []
        for yaw_quaternion in yaw_rotations:
            pose = position+ yaw_quaternion
            poses.append(pose)

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
        intermediary_pose = Pose(
            position=intermediary_position,
            quaternion=intermediary_orientation
        )
        # print(intermediary_pose)
        return intermediary_pose

    def set_constraint(self):
        self.plan_config.pose_cost_metric = self.linear_cost
        
    def release_constraint(self):
        self.plan_config.pose_cost_metric = self.freespace_cost
    def scale_velocity(self, scale):
        self.plan_config.time_dilation_factor = scale
        print("Setting Time Dilation Factor (Speed): ", self.plan_config.time_dilation_factor)


    def create_and_attach_object(self, ee_pos, initial_js) -> None:
        cu_js = JointState.from_position(
            position=self.tensor_args.to_device(initial_js),
            joint_names=self.j_names[0 : len(initial_js)],
        )

        # TODO: This function does not clear out previously attached objects and will cause memory leakage!
        # Can be avoided for now simply by calling this function sparingly
        if self.last_added_object is not None:
            self.world_cfg.remove_obstacle(self.last_added_object)
        # USE FK TO FIND WHERE TO PLACE OBJECT
        ee_pos = self.motion_gen.ik_solver.fk(cu_js.position).ee_position.squeeze().cpu().numpy()
        print("EE Position: ", ee_pos)
        object = Cuboid(
            name="object",
            pose=[ee_pos[0], ee_pos[1], ee_pos[2], 1.0, 0.0, 0.0, 0.0],
            dims=[0.1, 0.2, 0.055], # length, width, height (0.1, 0.2 are like shelf orientation)
            )
        
        
        self.world_cfg.add_obstacle(object)
        self.last_added_object = 'object'
        self.motion_gen.update_world(self.world_cfg)
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 20}

        self.motion_gen.detach_object_from_robot()
        self.motion_gen.attach_external_objects_to_robot(
            joint_state=cu_js,
            external_objects=[object],
            surface_sphere_radius=0.005,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
        )
        
    def detach_obj(self) -> None:
        self.detach = True
        self.motion_gen.detach_spheres_from_robot()

    def go_home(self,
                initial_js: List[float],
                goal_js: List[float]):
        if goal_js is not None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )        
            
            goal_js = JointState.from_position(
                position=self.tensor_args.to_device([goal_js]),
                joint_names=self.j_names[0 : len(goal_js)],
            )
            goal = Goal(goal_state=goal_js, current_state=initial_js)
            motion_gen_result = self.motion_gen.trajopt_solver.solve_any(solve_type=ReacherSolveType.SINGLE, goal=goal)
            reach_succ = motion_gen_result.success.item()
            print(self.motion_gen.compute_kinematics(goal_js).ee_pos_seq[0].cpu().numpy())
            print(self.motion_gen.compute_kinematics(goal_js).ee_quat_seq[0].cpu().numpy())

            if not reach_succ:
                print("Failed to reach goal")
                return None
            interpolated_solution = motion_gen_result.interpolated_solution


        else:
            raise ValueError("Check Goal JS")
        
        if reach_succ: 
            solution_dict = {
                "success": motion_gen_result.success.item(),
                "joint_names": interpolated_solution.joint_names,
                "positions": interpolated_solution.position.cpu().squeeze().numpy().tolist(),
                "velocities": interpolated_solution.velocity.cpu().squeeze().numpy().tolist(),
                "accelerations": interpolated_solution.acceleration.cpu().squeeze().numpy().tolist(),
                "jerks": interpolated_solution.jerk.cpu().squeeze().numpy().tolist(),
                "interpolation_dt": motion_gen_result.optimized_dt,
                "raw_data": interpolated_solution,
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
                interpolated_solution = motion_gen_result.get_interpolated_plan() 

            except Exception as e:
                print("Error in planning trajectory: ", e)
                return None
        elif goal_js_pose is not None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )        
            goal_js_pose = [1.3056849, 1.4040100, -0.34258141, 1.743283, 0.017052, 1.627947, -0.129718]

            goal_js = JointState.from_position(
                position=self.tensor_args.to_device([goal_js_pose]),
                joint_names=self.j_names[0 : len(goal_js_pose)],
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single_js(
                    initial_js, goal_js, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()
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
            print(" Cleared the world layer")

        data_camera = CameraObservation(rgb_image = camera_data["rgba"], depth_image=camera_data["depth"], intrinsics=camera_data["intrinsics"], 
                                        pose=camera_pose)
        print(" camera observation set ")
        self.world_model.add_camera_frame(data_camera, "world")
        self.world_model.process_camera_frames("world", process_aux=True)
        torch.cuda.synchronize()
        self.world_model.update_blox_hashes()

        print("update blox steps done")
        bounding = Cuboid("t", dims=[3, 3, 3.0], pose=[0, 0, 0, 1, 0, 0, 0])
        # bounding = Cuboid("t", dims=[2, 2, 2], pose=[0.5, 0.5, 0.5, 1, 0, 0, 0])

        try:
            voxels = self.world_model.get_voxels_in_bounding_box(bounding, self.voxel_size)
            # mesh = self.world_model.get_mesh_in_bounding_box(bounding, )
            # mesh = self.world_model.get_mesh_from_blox_layer(layer_name = "world", mode="voxel")
            print("Voxels before filtering:", voxels)
        except Exception as e:
            print("Error : ", e)
            voxels = None
        mesh = None
        
    
        if voxels.shape[0] > 0:
            voxels = voxels[voxels[:, 2] > self.voxel_size]
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


    # def generate_trajectory(self,
    #                    initial_js: List[float],
    #                    goal_ee_pose: List[float] = None,
    #                 #    goal_js: Optional[List[float]] = None,
    #                    suction_status: bool = False,
    # ):
    #     # if goal_ee_pose is not None and goal_js is None:
    #     if goal_ee_pose is not None:

    #         initial_js = JointState.from_position(
    #             position=self.tensor_args.to_device([initial_js]),
    #             joint_names=self.j_names[0 : len(initial_js)],
    #         )
    #         # initial_ee_pos = self.motion_gen.compute_kinematics(initial_js).ee_pos_seq[0].cpu().numpy()
            
    #         goal_pose = Pose(
    #             position=self.tensor_args.to_device(goal_ee_pose[0:3]),
    #             quaternion=self.tensor_args.to_device(goal_ee_pose[3:]),
    #         )
            
    #         if suction_status: # IF suction gripper is operating, initialize placing constraint (approach in z direction)
    #             # TODO: NEED TO GENERATE TWO TRAJECTORIES
    #             self.plan_config.pose_cost_metric = self.pulling_cost
    #             # ONE FOR PULLING A BOX OUT OF THE OBJECT (pose constraint everywhere except z)
    #             # OTHER FOR placing in the box (pose constraint on gripper orientation)
    #             # CONCATENATE THE TWO TRAJECTORIES AND RETURN
    #             # MAY CAUSE PROBLEM WHEN USED TO EXECUTE TRAJECTORIES

    #             # ALTERNATIVE:
    #                 # USE moveit1 for the pulling out part using cartesian planning and use curobo for the placing part
                
    #             ##################################### PULL OUT OF BOX #####################################
    #             # FIRST GOAL POSE IS +Z ABOVE THE INITIAL_JS
    #             ee_pose = self.motion_gen.ik_solver.fk(self.motion_gen.tensor_args.to_device(initial_js.position))
    #             ee_pos = ee_pose.ee_position
    #             ee_pos[0][2] += 0.20 # Move EE z direction up by 20cm
    #             ee_quat = ee_pose.ee_quaternion
    #             first_waypoint = Pose(
    #                 position=ee_pos,
    #                 quaternion=ee_quat,
    #             )
    #             try:
    #                 waypoint_result = self.motion_gen.plan_single(
    #                     initial_js, first_waypoint, self.plan_config
    #                 )
    #                 reach_succ = waypoint_result.success.item()
    #             except:
    #                 return None
    #             pulling_motion = waypoint_result.get_interpolated_plan()
    #             print(waypoint_result.get_interpolated_plan().position[-1].cpu().numpy())
    #             ##################################### PLACING IN THE BOX #####################################
    #             self.plan_config.pose_cost_metric = self.placing_cost

    #             try:
    #                 initial_js = JointState.from_position(
    #                     position=self.tensor_args.to_device([waypoint_result.get_interpolated_plan().position[-1].cpu().numpy()]),
    #                     joint_names=self.j_names[0 : len(initial_js)],
    #                 )

    #                 motion_gen_result = self.motion_gen.plan_single(
    #                     initial_js, goal_pose, self.plan_config
    #                 )
    #                 reach_succ = motion_gen_result.success.item()
    #             except:
    #                 return None
    #             waypoint_to_goal_result = motion_gen_result.get_interpolated_plan()
    #             # SOMETHING LIKE BELOW maybe need to debug
    #             # Concatenat`ing the positions from the two motions
    #             concatenated_positions = torch.cat(
    #                 (pulling_motion.position, waypoint_to_goal_result.position), dim=0
    #             )

    #             # Optionally, you can also concatenate velocities and other data if needed.
    #             concatenated_velocities = torch.cat(
    #                 (pulling_motion.velocity, waypoint_to_goal_result.velocity), dim=0
    #             )
    #             concatenated_acceleration = torch.cat(
    #                 (pulling_motion.acceleration, waypoint_to_goal_result.acceleration), dim=0
    #             )
    #             concatenated_jerks = torch.cat(
    #                 (pulling_motion.jerk, waypoint_to_goal_result.jerk), dim=0
    #             )
                
    #             # Create a new JointState with the concatenated positions
    #             final_joint_state = JointState(
    #                 position=concatenated_positions,
    #                 velocity=concatenated_velocities,
    #                 acceleration=concatenated_acceleration,
    #                 jerk=concatenated_jerks,
    #                 joint_names=self.j_names  # Ensure the joint names are correct
    #             )
    #             interpolated_solution = final_joint_state 

    #         else:   # IF suction is not operating, initialize picking cost (approach object in z direction)
    #             self.plan_config.pose_cost_metric = self.picking_cost

    #             try:
    #                 motion_gen_result = self.motion_gen.plan_single(
    #                     initial_js, goal_pose, self.plan_config
    #                 )
    #                 reach_succ = motion_gen_result.success.item()
    #                 interpolated_solution = motion_gen_result.get_interpolated_plan() 

    #             except Exception as e:
    #                 print("Error in planning trajectory: ", e)
    #                 return None
            
    #     # elif goal_js is not None and goal_ee_pose is None:
    #     #     initial_js = JointState.from_position(
    #     #         position=self.tensor_args.to_device([initial_js]),
    #     #         joint_names=self.j_names[0 : len(initial_js)],
    #     #     )        
            
    #     #     goal_js = JointState.from_position(
    #     #         position=self.tensor_args.to_device([goal_js]),
    #     #         joint_names=self.j_names[0 : len(goal_js)],
    #     #     )
            
    #     #     try:
    #     #         motion_gen_result = self.motion_gen.plan_single_js(
    #     #             initial_js, goal_js, self.plan_config
    #     #         )
    #     #         reach_succ = motion_gen_result.success.item()

    #     #     except:
    #     #         return None
    #     else:
    #         raise ValueError("Either goal_js or goal_ee_pose must be provided.")
        
    #     if reach_succ:
    #         # interpolated_solution = motion_gen_result.get_interpolated_plan() 
        
    #         solution_dict = {
    #             "success": motion_gen_result.success.item(),
    #             "joint_names": interpolated_solution.joint_names,
    #             "positions": interpolated_solution.position.cpu().squeeze().numpy().tolist(),
    #             "velocities": interpolated_solution.velocity.cpu().squeeze().numpy().tolist(),
    #             "accelerations": interpolated_solution.acceleration.cpu().squeeze().numpy().tolist(),
    #             "jerks": interpolated_solution.jerk.cpu().squeeze().numpy().tolist(),
    #             "interpolation_dt": motion_gen_result.interpolation_dt,
    #             "raw_data": interpolated_solution,
    #         }
            
    #         return solution_dict
        
    #     else:
    #         return None
        
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
