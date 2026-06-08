FROM osrf/ros:jazzy-desktop

# 1. System Upgrade & Core Development Tools
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    sudo nano vim git mesa-utils \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-pip \
    python3-numpy \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# 2. Simulation, Navigation, and Vision Libraries (For Capture the Flag)
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-slam-toolbox \
    ros-jazzy-robot-localization \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-topic-tools \
    && rm -rf /var/lib/apt/lists/*

# 3. Handle the existing UID 1000 user
ARG USERNAME=davi
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if id -u $USER_UID >/dev/null 2>&1; \
    then \
        existing_user=$(id -nu $USER_UID); \
        usermod -l $USERNAME $existing_user; \
        usermod -m -d /home/$USERNAME $USERNAME; \
        groupmod -n $USERNAME $(id -ng $USERNAME); \
    else \
        groupadd --gid $USER_GID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 4. Initialize Colcon Mixins for better build performance
RUN if ! colcon mixin list | grep -q 'default'; then \
        colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml; \
    fi && \
    colcon mixin update default

USER $USERNAME
WORKDIR /home/$USERNAME/robos_moveis

# 5. Inject the final .bashrc configurations directly during build
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "if [ -f ~/robos_moveis/install/setup.bash ]; then source ~/robos_moveis/install/setup.bash; fi" >> ~/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc && \
    echo "export _colcon_cd_root=~/robos_moveis" >> ~/.bashrc && \
    echo "alias cbuild='colcon build --symlink-install'" >> ~/.bashrc && \
    echo '# Auto-detect host X display from mounted socket so DISPLAY survives host-session changes' >> ~/.bashrc && \
    echo 'for s in /tmp/.X11-unix/X*; do [ -S "$s" ] && export DISPLAY=":${s##*/X}" && break; done' >> ~/.bashrc && \
    touch ~/.sudo_as_admin_successful

