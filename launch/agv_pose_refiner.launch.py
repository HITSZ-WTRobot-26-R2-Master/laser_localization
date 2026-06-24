from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _create_node(context):
    params = []
    shared = LaunchConfiguration('shared_params_file').perform(context)
    if shared:
        params.append(shared)
    params_file_val = LaunchConfiguration('params_file').perform(context)
    if params_file_val:
        params.append(params_file_val)

    return [
        Node(
            package="agv_pose_refiner",
            executable="agv_pose_refiner_node",
            name="agv_pose_refiner",
            output="screen",
            parameters=[
                {
                    "sensors_config_path": LaunchConfiguration("sensors_config_path"),
                    "solver_config_path": LaunchConfiguration("solver_config_path"),
                    "publish_tf": ParameterValue(
                        LaunchConfiguration("publish_tf"),
                        value_type=bool,
                    ),
                },
            ] + params,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("agv_pose_refiner")

    sensors_config_default = PathJoinSubstitution(
        [package_share, "config", "sensors.yaml"]
    )
    solver_config_default = PathJoinSubstitution(
        [package_share, "config", "map_and_solver.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sensors_config_path",
                default_value=sensors_config_default,
            ),
            DeclareLaunchArgument(
                "solver_config_path",
                default_value=solver_config_default,
            ),
            DeclareLaunchArgument(
                "publish_tf",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "shared_params_file",
                default_value="",
                description="Path to shared ROS 2 parameter YAML (e.g. config.yaml)",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="Path to service-specific ROS 2 parameter YAML.",
            ),
            OpaqueFunction(function=_create_node),
        ]
    )
