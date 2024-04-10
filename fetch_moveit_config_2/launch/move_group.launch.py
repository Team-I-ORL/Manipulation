from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("fetch", package_name="fetch_moveit_config_2")
        .robot_description(
            file_path="config/fetch.urdf.xacro",
            # mappings={
            #     "ros2_control_hardware_type": LaunchConfiguration(
            #         "ros2_control_hardware_type"
            #     )
            # },
        )
        .robot_description_semantic(file_path="config/fetch.srdf")
        .planning_scene_monitor(
           publish_robot_description=True, publish_robot_description_semantic=True,
           publish_geometry_updates= True, publish_state_updates= True, 
           publish_transforms_updates= True
        )
        # #.moveit_cpp(file_path="/home/siddharth/ws_moveit/src/Manipulation/fetch_moveit_config/config/motion_planning_python_api_tutorial.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner", "stomp"]
        )
        .to_moveit_configs()
    )

    # moveit_config = MoveItConfigsBuilder("fetch", package_name="fetch_moveit_config_2").to_moveit_configs()
    return generate_move_group_launch(moveit_config)
