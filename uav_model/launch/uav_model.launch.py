from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution(
            [FindPackageShare('uav_model'), 'config', 'params.yaml']
        ),
        description='Full path to the ROS2 parameter file for uav_model_node',
    )

    uav_model_node = Node(
        package='uav_model',
        executable='uav_model_node',
        name='uav_model_node',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('thrust_and_torque', '/cf_control/control_command'),
            ('state', '/uav_model/state'),
        ],
        output='screen',
    )

    return LaunchDescription([
        params_file_arg,
        uav_model_node,
    ])
