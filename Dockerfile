FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    curl \
    git \
    libeigen3-dev \
    less \
    locales \
    nano \
    python3-dev \
    python3-pip \
    python3-venv \
    software-properties-common \
    sudo \
    unzip \
    wget \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
 && add-apt-repository -y universe \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') \
 && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
 && dpkg -i /tmp/ros2-apt-source.deb \
 && rm /tmp/ros2-apt-source.deb \
 && apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y \
    ros-dev-tools \
    ros-humble-ackermann-msgs \
    ros-humble-desktop \
    ros-humble-foxglove-bridge \
    ros-humble-nav2-amcl \
    ros-humble-nav2-bringup \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-nav2-map-server \
    ros-humble-nav2-util \
    ros-humble-slam-toolbox \
    ros-humble-xacro \
 && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
 && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /workspace

CMD ["bash"]
