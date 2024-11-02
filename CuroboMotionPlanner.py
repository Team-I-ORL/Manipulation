import torch
# CuRobo
# Standard Library
from typing import List, Optional

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

        world_cfg = WorldConfig(cuboid=mobile_base_cfg.cuboid)
        # world_cfg = WorldConfig.create_obb_world(world_cfg)
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
            project_pose_to_goal_frame=True,
            minimize_jerk=True,
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
            'partial_ik_opt': False,
            'parallel_finetune': True,
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
            collision_cache={"obb": 200, "mesh": 100, "voxels":100},
            trajopt_tsteps=24,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=True,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.01,
            collision_activation_distance=0.025,
            acceleration_scale=1.0,
            self_collision_check=True,
            maximum_trajectory_dt=0.25,
            finetune_dt_scale=1.05,
            fixed_iters_trajopt=True,
            finetune_trajopt_iters=300,
            minimize_jerk=True,
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

    def setup_ik_solver(self):
        ik_solver_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            num_seeds=20,
        )
        ik_solver= IKSolver(ik_solver_config)

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
        print(intermediary_pose)
        return intermediary_pose

    def set_constraint(self):
        self.plan_config.pose_cost_metric = self.linear_cost
        
    def release_constraint(self):
        self.plan_config.pose_cost_metric = self.freespace_cost
    def scale_velocity(self, scale):
        self.plan_config.time_dilation_factor = scale
        print("Setting Time Dilation Factor (Speed): ", self.plan_config.time_dilation_factor)

    def generate_trajectory(self,
                       initial_js: List[float],
                       goal_ee_pose: List[float] = None,
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
                print(motion_gen_result)
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
    def update_blox_from_camera(self, camera_data, camera_pose) -> None:
        self.world_model.decay_layer("world")
        data_camera = CameraObservation(depth_image=camera_data["depth"], intrinsics=camera_data["intrinsics"], 
                                        pose=camera_pose)
        
        self.world_model.add_camera_frame(data_camera, "world")
        self.world_model.process_camera_frames("world", False)
        torch.cuda.synchronize()
        self.world_model.update_blox_hashes()

        bounding = Cuboid("t", dims=[2, 2, 2.0], pose=[0, 0, 0, 1, 0, 0, 0])
        # bounding = Cuboid("t", dims=[2, 2, 2], pose=[0.5, 0.5, 0.5, 1, 0, 0, 0])


        print("Depth Image (sample):", camera_data["depth"][0:5, 0:5])  # Sample depth values
        print("Bounding Box:", bounding)

        voxels = self.world_model.get_voxels_in_bounding_box(bounding, self.voxel_size)
        print("Voxels before filtering:", voxels)
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
            color_image = camera_data["raw_rgb"].cpu().numpy()
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

        return voxels    
    
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
