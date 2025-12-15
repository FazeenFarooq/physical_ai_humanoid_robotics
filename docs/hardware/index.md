# Hardware Setup and Troubleshooting

<content>

Welcome to the hardware documentation for the Physical AI & Humanoid Robotics Course. This section provides detailed information about setting up and troubleshooting the various hardware components used in the course.

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [System Requirements](#system-requirements)
3. [Jetson Orin Setup](#jetson-orin-setup)
4. [Robot Platform Setup](#robot-platform-setup)
5. [Simulation Workstation Setup](#simulation-workstation-setup)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Safety Guidelines](#safety-guidelines)

## Hardware Overview

The Physical AI course utilizes several types of hardware resources:

- **NVIDIA Jetson Orin Development Kits**: For edge AI processing and robot control
- **Humanoid Robot Platforms**: Unitree H1, TurtleBot 4, or similar platforms
- **Simulation Workstations**: High-performance Linux workstations with RTX GPUs
- **Sensors**: RGB-D cameras, LiDAR, IMU, etc.
- **Network Infrastructure**: For communication between components

## System Requirements

### NVIDIA Jetson Orin
- **Model**: NVIDIA Jetson AGX Orin (64GB)
- **CPU**: 8-core ARM v8.4 64-bit CPU
- **GPU**: 2048-core NVIDIA Ampere GPU
- **Memory**: 64GB LPDDR5
- **Storage**: 64GB eMMC
- **OS**: JetPack 5.1+ with ROS 2 Humble Hawksbill

### Simulation Workstation
- **CPU**: AMD Ryzen 7 7800X3D or Intel i7-13700K
- **GPU**: NVIDIA RTX 4090 with 24GB+ VRAM
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD
- **OS**: Ubuntu 22.04 LTS
- **Network**: Gigabit Ethernet or better

## Jetson Orin Setup

### Initial Setup
1. Flash the JetPack 5.1+ image to the Jetson Orin device
2. Connect power, monitor, keyboard, and mouse
3. Complete initial OS setup, creating a user account
4. Connect to the internet via Ethernet or WiFi

### Software Installation
1. Update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. Install ROS 2 Humble Hawksbill:
   ```bash
   sudo apt update && sudo apt install -y \
     ros-humble-desktop \
     ros-humble-cv-bridge \
     ros-humble-tf2-tools \
     ros-humble-nav2-bringup
   ```

3. Install Python dependencies:
   ```bash
   pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install transformers openai gymnasium numpy pandas matplotlib opencv-python
   ```

4. Source ROS environment:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Network Configuration
1. Configure the ROS domain ID by setting environment variable:
   ```bash
   echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
   ```

2. Set a static IP address for consistent communication with other devices

## Robot Platform Setup

### Unitree H1 Specific Setup
1. Verify physical connections as per Unitree documentation
2. Update robot firmware to the latest version
3. Configure communication parameters (baud rate, protocol, etc.)
4. Install Unitree SDK and ROS packages

### Safety Precautions
1. Always perform initial tests with the robot restrained or suspended
2. Ensure emergency stop procedures are accessible and tested
3. Verify that safety limits are properly configured
4. Test all sensors and emergency protocols before autonomous operation

## Simulation Workstation Setup

### Initial Configuration
1. Install Ubuntu 22.04 LTS
2. Install NVIDIA drivers and CUDA toolkit
3. Configure GPU settings for optimal performance

### ROS 2 Setup
1. Install ROS 2 Humble Hawksbill following the official installation guide
2. Set up your ROS workspace:
   ```bash
   mkdir -p ~/ai_robotics_ws/src
   cd ~/ai_robotics_ws
   colcon build
   source install/setup.bash
   ```

### Simulation Environment
1. Install Gazebo Garden:
   ```bash
   sudo apt install -y gz-fortress
   ```

2. Install NVIDIA Isaac Sim (optional, for advanced simulation):
   - Download from NVIDIA Developer website
   - Follow installation instructions specific to your GPU

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Jetson Orin overheating
- **Symptoms**: Performance throttling, shutdown, high temperature readings
- **Causes**: Inadequate cooling, intensive processing, poor ventilation
- **Solutions**:
  1. Check that cooling fans are operational
  2. Use active cooling solutions if in an enclosure
  3. Reduce computational load temporarily
  4. Monitor temperature with `sudo tegrastats`

#### Issue: Communication problems between devices
- **Symptoms**: Nodes not connecting, timeout errors, no topic exchange
- **Causes**: Network configuration issues, firewall settings, ROS domain mismatches
- **Solutions**:
  1. Verify all devices are on the same subnet
  2. Check `ROS_DOMAIN_ID` environment variable on all devices
  3. Test basic connectivity with `ping`
  4. Verify firewall settings allow ROS traffic

#### Issue: Camera or sensor not detected
- **Symptoms**: No camera feed, sensor data not publishing
- **Causes**: Driver issues, USB connection problems, incorrect configuration
- **Solutions**:
  1. Check physical connections
  2. Verify device permissions: `lsusb` or `v4l2-ctl --list-devices`
  3. Check driver installation: `dmesg | grep -i usb`
  4. Test with basic tools like `v4l2-ctl --device=0 --stream-mmap --stream-count=1`

#### Issue: Robot motion not smooth or unstable
- **Symptoms**: Jerky movements, oscillation, inaccurate positioning
- **Causes**: Control parameters not tuned, sensor noise, mechanical issues
- **Solutions**:
  1. Calibrate sensors (IMU, encoders, etc.)
  2. Tune PID controllers
  3. Check for mechanical backlash or wear
  4. Verify that control loop timing is consistent

### Diagnostic Tools

#### System Monitoring
- For Jetson devices: `sudo tegrastats` for real-time performance monitoring
- For all systems: `htop` or `top` for CPU/memory usage
- Network: `nethogs` to identify bandwidth usage by process

#### ROS-Specific Tools
- `ros2 topic list` and `ros2 topic echo` to verify topic communication
- `rqt_graph` to visualize the ROS computation graph
- `ros2 lifecycle` to check node lifecycle states

## Safety Guidelines

### General Safety
- Always maintain an emergency stop procedure that can quickly stop robot motion
- Ensure adequate lighting and clear workspace before operating robots
- Never operate robots unsupervised for extended periods
- Keep emergency contact information and first aid supplies accessible

### Electrical Safety
- Verify all electrical connections before powering on hardware
- Use appropriate power supplies for each component
- Check for proper grounding of all equipment
- Be aware of high-voltage components and follow lockout/tagout procedures when appropriate

### Physical Safety
- Maintain safe distances from moving robot parts
- Ensure robot workspace is clear of obstacles and people
- Use appropriate barriers or safety zones during testing
- Always approach robots from the front with caution

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-developer-kits)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Unitree Robotics Support](https://www.unitree.com/support/)
- Course-specific configurations in `configs/deployment/jetson_config.yaml`

</content>