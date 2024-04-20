import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os 

def generate_launch_description():
    
    #pkg_path = launch_ros.substitutions.FindPackage(package='fetch_description').find('fetch_description')
    #print(pkg_path)
    #urdf_model_path = os.path.join(pkg_path, 'urdf/fetch.urdf')
    urdf_model_path = '/home/siddharth/ws_moveit/src/Manipulation/fetch_description/urdf/fetch.urdf'
    rviz_path = '/home/siddharth/ws_moveit/src/fetch_description/rviz/display_rviz.rviz'

    with open(urdf_model_path, 'r') as infp:
        robot_desc = infp.read()

    params = {'robot_description': robot_desc}

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[params],
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('gui'))
    )

    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d' + rviz_path]    
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True', description='Flag for join_state_gui'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=urdf_model_path, description='path to urdf'),
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
