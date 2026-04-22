"""
Launch the Mellinger controller in one of two modes.

Mode A — simulation (default):
    Controller + uav_model_node form a closed loop.  No Gazebo required.
    State feedback comes from uav_model_node (/uav_model/state).

    ros2 launch mellinger_controller mellinger.launch.py

Mode B — Gazebo-in-the-loop:
    Controller reads state directly from Gazebo odometry and sends thrust
    commands to the cf_control mixer, which drives Gazebo motors.
    uav_model_node is NOT started.

    ros2 launch mellinger_controller mellinger.launch.py use_gazebo:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    use_gazebo = LaunchConfiguration('use_gazebo').perform(context).lower() in ('true', '1', 'yes')
    mellinger_params = LaunchConfiguration('mellinger_params_file').perform(context)
    uav_params = LaunchConfiguration('uav_params_file').perform(context)

    if use_gazebo:
        # Mode B: Gazebo is the plant — read odometry from Gazebo, write to mixer.
        # Gazebo OdometryPublisher publishes twist in the child (body) frame.
        state_topic = '/crazyflie/odom'
        cmd_topic = '/cf_control/control_command'
    else:
        # Mode A: uav_model_node is the plant — closed simulation loop.
        state_topic = '/uav_model/state'
        cmd_topic = '/cf_control/control_command'

    mellinger_node = Node(
        package='mellinger_controller',
        executable='mellinger_controller',
        name='mellinger_controller',
        parameters=[mellinger_params],
        remappings=[
            ('state', state_topic),
            ('thrust_and_torque', cmd_topic),
            ('waypoints', '/mellinger/waypoints'),
        ],
        output='screen',
    )

    nodes = [mellinger_node]

    if not use_gazebo:
        uav_model_node = Node(
            package='uav_model',
            executable='uav_model_node',
            name='uav_model_node',
            parameters=[uav_params],
            remappings=[
                ('thrust_and_torque', cmd_topic),
                ('state', '/uav_model/state'),
            ],
            output='screen',
        )
        nodes.append(uav_model_node)

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='false',
            description='If true, read state from /crazyflie/odom and skip uav_model_node',
        ),
        DeclareLaunchArgument(
            'mellinger_params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('mellinger_controller'), 'config',
                 'mellinger_params.yaml']
            ),
            description='Full path to the Mellinger controller parameter file',
        ),
        DeclareLaunchArgument(
            'uav_params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('uav_model'), 'config', 'params.yaml']
            ),
            description='Full path to the UAV model parameter file (Mode A only)',
        ),
        OpaqueFunction(function=_make_nodes),
    ])
