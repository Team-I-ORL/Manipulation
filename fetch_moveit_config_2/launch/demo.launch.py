from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("fetch", package_name="fetch_moveit_config_2").robot_description(
            file_path="config/fetch.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": 'isaac'
            },
        ).to_moveit_configs()
    return generate_demo_launch(moveit_config)
