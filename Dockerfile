# Build from this package directory:
#   docker build -t laser_localization:humble .

ARG BASE_IMAGE=r2_master_interface:humble
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=humble
ARG UBUNTU_MIRROR=https://mirror.nju.edu.cn/ubuntu/
ARG ROS2_APT_MIRROR=https://mirrors.nju.edu.cn/ros2/ubuntu

ENV ROS_DISTRO=${ROS_DISTRO} \
    WORKSPACE_DIR=/workspace \
    LASER_LOCALIZATION_DIR=/workspace/src/positioning/laser_localization \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN <<EOF
set -euo pipefail
. /etc/os-release
codename="${VERSION_CODENAME}"
rm -f /etc/apt/sources.list.d/ubuntu.sources
cat >/etc/apt/sources.list <<SOURCES
# Source mirror entries are commented out to keep apt update fast.
deb ${UBUNTU_MIRROR} ${codename} main restricted universe multiverse
# deb-src ${UBUNTU_MIRROR} ${codename} main restricted universe multiverse
deb ${UBUNTU_MIRROR} ${codename}-updates main restricted universe multiverse
# deb-src ${UBUNTU_MIRROR} ${codename}-updates main restricted universe multiverse
deb ${UBUNTU_MIRROR} ${codename}-backports main restricted universe multiverse
# deb-src ${UBUNTU_MIRROR} ${codename}-backports main restricted universe multiverse

# deb ${UBUNTU_MIRROR} ${codename}-security main restricted universe multiverse
# deb-src ${UBUNTU_MIRROR} ${codename}-security main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ ${codename}-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ ${codename}-security main restricted universe multiverse

# Prerelease repository, not recommended.
# deb ${UBUNTU_MIRROR} ${codename}-proposed main restricted universe multiverse
# deb-src ${UBUNTU_MIRROR} ${codename}-proposed main restricted universe multiverse
SOURCES
for ros_source in \
    /etc/apt/sources.list.d/ros2.sources \
    /etc/apt/sources.list.d/ros2.list \
    /etc/apt/sources.list.d/ros2-latest.list \
    /usr/share/ros-apt-source/ros2.sources; do
    if [ -e "${ros_source}" ] || [ -L "${ros_source}" ]; then
        sed -i --follow-symlinks \
            -e "s#http://packages.ros.org/ros2/ubuntu#${ROS2_APT_MIRROR}#g" \
            -e "s#https://packages.ros.org/ros2/ubuntu#${ROS2_APT_MIRROR}#g" \
            -e "s#^Types: deb deb-src\$#Types: deb#g" \
            "${ros_source}"
    fi
done
EOF

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        procps \
        python3-colcon-common-extensions \
        python3-serial \
        python3-setuptools \
        python3-yaml \
        ros-${ROS_DISTRO}-ament-cmake \
        ros-${ROS_DISTRO}-ament-cmake-pytest \
        ros-${ROS_DISTRO}-ament-cmake-python \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-launch \
        ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-rclpy \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-tf2-ros

WORKDIR ${WORKSPACE_DIR}

COPY CMakeLists.txt \
     package.xml \
     setup.cfg \
     setup.py \
     src/positioning/laser_localization/
COPY agv_pose_refiner_py/ src/positioning/laser_localization/agv_pose_refiner_py/
COPY config/ src/positioning/laser_localization/config/
COPY launch/ src/positioning/laser_localization/launch/
COPY resource/ src/positioning/laser_localization/resource/
COPY scripts/ src/positioning/laser_localization/scripts/
COPY test/ src/positioning/laser_localization/test/

RUN echo '# placeholder - overridden at runtime by docker-compose volume mount' \
       > src/positioning/laser_localization/config/agv_pose_refiner.yaml

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && source "${WORKSPACE_DIR}/install/setup.bash" \
    && colcon build \
        --packages-select agv_pose_refiner \
        --symlink-install \
    && test -x "${WORKSPACE_DIR}/install/agv_pose_refiner/lib/agv_pose_refiner/agv_pose_refiner_node" \
    && test -f "${WORKSPACE_DIR}/install/agv_pose_refiner/share/agv_pose_refiner/config/topics.yaml" \
    && test -f "${WORKSPACE_DIR}/install/agv_pose_refiner/share/agv_pose_refiner/config/sensors.yaml" \
    && test -f "${WORKSPACE_DIR}/install/agv_pose_refiner/share/agv_pose_refiner/config/map_and_solver.yaml" \
    && ros2 interface show interfaces/msg/R2Pose >/dev/null

RUN mkdir -p "${LASER_LOCALIZATION_DIR}/config" \
    && rm -rf "${WORKSPACE_DIR}/install/agv_pose_refiner/share/agv_pose_refiner/config" \
    && ln -s "${LASER_LOCALIZATION_DIR}/config" "${WORKSPACE_DIR}/install/agv_pose_refiner/share/agv_pose_refiner/config"

VOLUME ["/workspace/src/positioning/laser_localization/config"]

COPY --chmod=755 docker/ros_entrypoint.sh /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "agv_pose_refiner", "agv_pose_refiner.launch.py"]
