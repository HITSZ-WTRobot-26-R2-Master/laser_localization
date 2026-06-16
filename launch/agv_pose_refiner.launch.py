from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("agv_pose_refiner")

    topics_config_default = PathJoinSubstitution(
        [package_share, "config", "topics.yaml"]
    )
    sensors_config_default = PathJoinSubstitution(
        [package_share, "config", "sensors.yaml"]
    )
    solver_config_default = PathJoinSubstitution(
        [package_share, "config", "map_and_solver.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "topics_config_path",
                default_value=topics_config_default,
            ),
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
            Node(
                package="agv_pose_refiner",
                executable="agv_pose_refiner_node",
                name="agv_pose_refiner",
                output="screen",
                parameters=[
                    {
                        "topics_config_path": LaunchConfiguration("topics_config_path"),
                        "sensors_config_path": LaunchConfiguration("sensors_config_path"),
                        "solver_config_path": LaunchConfiguration("solver_config_path"),
                        "publish_tf": ParameterValue(
                            LaunchConfiguration("publish_tf"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
