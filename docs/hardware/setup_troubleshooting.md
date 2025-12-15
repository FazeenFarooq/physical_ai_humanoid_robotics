# Hardware Setup and Troubleshooting Guide

This document provides instructions for setting up and troubleshooting the hardware used in the Physical AI & Humanoid Robotics course.

## Table of Contents
1. [Hardware Overview](#hardware-overview)
2. [Workstation Setup](#workstation-setup)
3. [Jetson Orin Setup](#jetson-orin-setup)
4. [Robot Platform Setup](#robot-platform-setup)
5. [Network Configuration](#network-configuration)
6. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
7. [Safety Protocols](#safety-protocols)
8. [Maintenance Schedule](#maintenance-schedule)

## Hardware Overview

The course utilizes three primary hardware tiers:

### Digital Twin Workstation
- **CPU**: AMD Ryzen 7 7800X3D or Intel i7-13700K (or equivalent)
- **GPU**: NVIDIA RTX 4090 with 24GB+ VRAM (minimum requirement)
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Edge AI Kit: NVIDIA Jetson AGX Orin
- **GPU**: NVIDIA Ampere architecture with 2048 CUDA cores
- **CPU**: ARM A78AE 8-core
- **Memory**: 32GB LPDDR5
- **Storage**: 32GB eMMC + microSD slot
- **OS**: JetPack/Linux for Tegra (L4T)

### Robot Lab Options
- **Humanoid**: Unitree H1 (or equivalent)
- **Wheeled**: TurtleBot 4 (or equivalent)
- **Custom platforms** with compatible ROS 2 drivers

## Workstation Setup

### Prerequisites
Ensure your workstation meets the specified requirements before proceeding.

### Step 1: Install Ubuntu 22.04 LTS
1. Download Ubuntu 22.04 LTS ISO from [official website](https://ubuntu.com/download/desktop)
2. Create a bootable USB using Rufus (Windows) or Startup Disk Creator (Ubuntu)
3. Boot from USB and follow installation instructions
4. During installation, select "Minimal installation" to reduce bloatware

### Step 2: System Updates and Essential Tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential cmake git wget curl htop iotop python3-dev python3-pip
```

### Step 3: Install NVIDIA Drivers and CUDA
1. Check GPU compatibility:
```bash
lspci | grep -i nvidia
```

2. Install NVIDIA driver:
```bash
sudo apt install nvidia-driver-535  # or latest compatible version
sudo reboot
```

3. Install CUDA toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```
> **Note**: During CUDA installation, deselect the driver if you already installed it separately.

### Step 4: Install Python Environment
```bash
# Install Python virtual environment tools
pip3 install --user pip virtualenv

# Create virtual environment
python3 -m venv ~/ai_robotics_env
source ~/ai_robotics_env/bin/activate

# Install Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers openai gymnasium numpy pandas matplotlib
pip install opencv-contrib-python open3d
```

### Step 5: Install ROS 2 Humble Hawksbill
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

### Step 6: Install Simulation Environment (Gazebo Garden)
```bash
# Install Gazebo Garden
sudo apt install -y gz-fortress

# Verify installation
gz --version
```

### Step 7: Install NVIDIA Isaac Sim (Optional for Advanced Simulation)
1. Register for NVIDIA Developer account at [developer.nvidia.com](https://developer.nvidia.com)
2. Download Isaac Sim from the NVIDIA Developer website
3. Follow the installation instructions provided with Isaac Sim
4. Ensure CUDA 12.2 and compatible drivers are installed

## Jetson Orin Setup

### Prerequisites
- NVIDIA Jetson AGX Orin Developer Kit
- Power adapter and USB-C cable
- MicroSD card (if needed for additional storage)
- Ethernet connection to the internet

### Step 1: Flash Jetson OS
1. Download NVIDIA SDK Manager from [developer.nvidia.com](https://developer.nvidia.com/nvidia-sdk-manager)
2. Install SDK Manager on a host machine with Ubuntu 18.04 or 20.04
3. Connect Jetson to host via USB-C
4. Run SDK Manager and select Jetson AGX Orin target
5. Follow wizard to flash OS to Jetson

### Step 2: Initial Setup
1. Connect keyboard, mouse, and monitor to Jetson
2. Power on Jetson and complete initial setup (locale, user account)
3. Connect to internet using Ethernet or WiFi

### Step 3: Install ROS 2 Humble (via Debian packages)
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install -y ros-humble-ros-base
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install Course Materials
1. Clone the course repository:
```bash
cd ~/
git clone https://github.com/course-org/physical-ai-course.git
```

2. Build the workspace:
```bash
cd ~/physical-ai-course
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Step 5: Optimize for Jetson Platform
1. Install TensorRT for optimized inference:
```bash
sudo apt install nvinfer-runtime-trtexec
```

2. Set performance mode:
```bash
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Lock clocks to maximum frequency
```

## Robot Platform Setup

### Unitree H1 Setup (Humanoid Platform)
1. Ensure robot battery is charged
2. Power on the robot
3. Connect to the robot's ROS 2 network
4. Follow the specific setup instructions provided with the robot

### TurtleBot 4 Setup (Wheeled Platform)
1. Ensure robot battery is charged
2. Power on the robot
3. Connect to the robot's ROS 2 network
4. Install TurtleBot 4 packages:
```bash
sudo apt install ros-humble-turtlebot4-*
```

## Network Configuration

### ROS 2 Network Setup
All devices need to be on the same network to communicate via ROS 2.

1. Ensure all devices have static IPs or reserved DHCP addresses
2. Set ROS_DOMAIN_ID consistently across all devices:
```bash
export ROS_DOMAIN_ID=42  # Use same domain ID on all devices
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
```

3. Configure firewall to allow ROS 2 communication:
```bash
# Allow necessary ports
sudo ufw allow 11311  # ROS master
sudo ufw allow 50051  # gRPC (if used)
sudo ufw allow 8080   # Web interfaces
sudo ufw allow 9090   # rosbridge (if used)
```

### Example Network Configuration
```
Workstation: 192.168.1.10
Jetson: 192.168.1.11
Robot: 192.168.1.12
```

## Common Issues and Troubleshooting

### GPU/CUDA Issues

**Issue**: CUDA not detected or PyTorch not using GPU
- **Solution**: 
  1. Verify NVIDIA driver: `nvidia-smi`
  2. Check CUDA installation: `nvcc --version`
  3. Test PyTorch GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`

**Issue**: CUDA version mismatch
- **Solution**: Ensure PyTorch version matches CUDA version. For CUDA 12.1, use the specific PyTorch install command.

### ROS 2 Issues

**Issue**: ROS 2 nodes cannot communicate between devices
- **Solution**:
  1. Check ROS_DOMAIN_ID consistency across devices
  2. Verify firewall settings
  3. Test network connectivity with ping
  4. Check ROS 2 daemon: `ros2 daemon status`

**Issue**: No ROS 2 nodes found
- **Solution**: Start ROS 2 daemon: `ros2 daemon start`

### Simulation Issues

**Issue**: Gazebo crashes or runs slowly
- **Solution**:
  1. Check graphics drivers
  2. Reduce simulation complexity
  3. Ensure sufficient RAM and VRAM availability

**Issue**: Isaac Sim fails to launch
- **Solution**:
  1. Verify CUDA compatibility
  2. Check system requirements are met
  3. Run with graphics debugger: `isaac-sim -- --verbose`

### Hardware-Specific Issues

**Jetson Overheating**
- **Solution**: 
  1. Ensure proper cooling setup
  2. Reduce performance demands temporarily
  3. Monitor temperature: `sudo tegrastats`

**Robot Communication Issues**
- **Solution**:
  1. Check network connectivity
  2. Verify ROS 2 domain IDs match
  3. Check robot-specific documentation

## Safety Protocols

### Before Operating Physical Hardware
- Ensure safety zones are established around the robot
- Have emergency stop procedures ready
- Verify all safety checks are passing

### During Operation
- Maintain clear communication with team members
- Monitor robot behavior continuously
- Be ready to activate emergency stop if needed

### Emergency Procedures
- If robot behaves unexpectedly, immediately activate emergency stop
- If hardware fails, power down safely
- Document all incidents for review

## Maintenance Schedule

### Daily Checks
- Verify all hardware components are present and undamaged
- Check battery levels of all portable devices
- Ensure network connectivity between components

### Weekly Checks
- Update system software (as scheduled)
- Check and clean sensors (if needed)
- Verify calibration of critical components

### Monthly Checks
- Deep system diagnostics
- Backup and archival of data
- Hardware maintenance as per manufacturer guidelines

### Troubleshooting Checklist

For any hardware issue, follow this checklist:

- [ ] Verify power connections
- [ ] Check network connectivity
- [ ] Confirm ROS 2 domain IDs match
- [ ] Check system resource usage (CPU, memory, GPU)
- [ ] Look for error messages in logs
- [ ] Restart relevant services if needed
- [ ] Verify calibration is current
- [ ] Check safety system status
- [ ] Document issue and resolution for future reference