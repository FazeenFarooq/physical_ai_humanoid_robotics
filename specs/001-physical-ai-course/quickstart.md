# Quickstart Guide: Physical AI & Humanoid Robotics Course

**Feature**: 001-physical-ai-course
**Date**: 2025-12-14

## Getting Started

This quickstart guide provides a rapid introduction to setting up your development environment and running your first Physical AI project. Follow these steps to get from zero to your first working robot system quickly.

## Prerequisites

Before starting, ensure your workstation meets these requirements:
- **Operating System**: Ubuntu 22.04 LTS (recommended) or other Linux distribution
- **CPU**: AMD Ryzen 7 7800X3D or Intel i7-13700K (or equivalent)
- **GPU**: NVIDIA RTX 4090 with 24GB+ VRAM (minimum requirement)
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD
- **Network**: Stable internet connection for package downloads

## Environment Setup

### 1. Install ROS 2 Humble Hawksbill

Open a terminal and run:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-cv-bridge ros-humble-tf2-tools ros-humble-nav2-bringup
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Simulation Environment (Gazebo Garden)

```bash
# Install Gazebo Garden
sudo apt install -y gz-fortress

# Verify installation
gz --version
```

### 3. Install NVIDIA Isaac Sim (Optional for Advanced Simulation)

If you need NVIDIA Isaac Sim for high-fidelity sensor simulation:

```bash
# Download Isaac Sim from NVIDIA Developer website
# Follow the installation instructions on the NVIDIA website
# Ensure CUDA 12.2 and compatible drivers are installed
```

### 4. Set Up Python Environment

```bash
# Install Python dependencies
sudo apt install -y python3-pip python3-dev
pip3 install --user pip virtualenv

# Create virtual environment
cd ~/
python3 -m venv ~/ai_robotics_env
source ~/ai_robotics_env/bin/activate

# Install Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers openai gymnasium numpy pandas matplotlib
pip install opencv-contrib-python open3d
```

### 5. Install Additional Tools

```bash
# Install Jetson Deployment Tools
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
./install.sh

# Install navigation and manipulation libraries
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-moveit ros-humble-moveit-visual-tools
```

### 6. Validate Installation

Test your installation with a simple ROS 2 example:

```bash
# Open a new terminal and source ROS
source /opt/ros/humble/setup.bash

# Run the talker node
ros2 run demo_nodes_cpp talker
```

In another terminal:

```bash
# Source ROS and run listener
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

You should see messages passing from talker to listener.

## First Simulation Exercise

### Step 1: Create a Workspace

```bash
# Create workspace directory
mkdir -p ~/ai_robotics_ws/src
cd ~/ai_robotics_ws

# Source ROS and build workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select demo_nodes_cpp
source install/setup.bash
```

### Step 2: Launch a Basic Robot Simulation

```bash
# Launch TurtleBot 4 simulation
ros2 launch turtlebot4_gz_bringup turtlebot4_ignition.launch.py
```

### Step 3: Control the Robot

Open another terminal and send a movement command:

```bash
# Source ROS environment
source /opt/ros/humble/setup.bash

# Send velocity command to robot
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
```

The robot should start moving in a circular pattern in the simulation.

## Course-Specific Setup

### 1. Clone Course Materials

```bash
cd ~/ai_robotics_ws/src
git clone https://github.com/course-org/physical-ai-course.git
cd ~/ai_robotics_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### 2. Run Course-Specific Example

```bash
# Launch Week 1 example - basic ROS 2 publisher/subscriber
cd ~/ai_robotics_ws
source install/setup.bash
ros2 run course_examples week1_basic_nodes
```

## Troubleshooting Common Issues

### GPU/CUDA Issues
- Verify CUDA installation: `nvidia-smi`
- Check CUDA version: `nvcc --version`
- Ensure PyTorch is using GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### ROS 2 Issues
- Ensure proper sourcing: `source /opt/ros/humble/setup.bash`
- Check ROS domain ID: `echo $ROS_DOMAIN_ID`
- Check ROS environment: `printenv | grep ROS`

### Simulation Issues
- Confirm Gazebo installation: `gz --version`
- Check for conflicting Gazebo versions
- Verify graphics drivers support hardware acceleration

### Network/Internet Issues
- Check proxy settings if behind corporate firewall
- Verify package repositories are accessible
- Consider using mirrors if download speed is slow

## Next Steps

After completing this quickstart guide:

1. **Week 1**: Complete the ROS 2 fundamentals exercises
2. **Week 2**: Set up your first custom robot model in simulation
3. **Week 3**: Integrate perception systems with your robot
4. **Week 4**: Begin development of your capstone project components

For more detailed instructions and troubleshooting, refer to the full course documentation in the `docs/` directory of your workspace.