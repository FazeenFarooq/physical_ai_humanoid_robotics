# Digital Twin Lab Exercises

This document outlines the lab exercises for the Digital Twin (Gazebo + Unity) module of the Physical AI & Humanoid Robotics course.

## Lab Exercise 1: Basic Gazebo Environment Setup

### Objective
Students will set up a basic Gazebo environment and spawn a robot model.

### Prerequisites
- ROS 2 environment setup
- Basic understanding of URDF models

### Steps
1. Install Gazebo Garden on your development machine
2. Verify Gazebo installation with basic commands
3. Create a custom Gazebo world with basic obstacles
4. Spawn a pre-defined robot model in the world
5. Control the robot using keyboard commands

### Resources Required
- Linux development machine with ROS 2
- Gazebo Garden installation
- Robot URDF model (provided by course)

### Validation Criteria
- Successfully launch Gazebo with a custom world
- Robot model spawns without errors
- Robot responds to basic control commands
- Environment includes static obstacles

---

## Lab Exercise 2: Custom Robot Model Integration

### Objective
Students will integrate their own robot model into the Gazebo simulation environment.

### Prerequisites
- Lab Exercise 1 completed
- Understanding of URDF/XACRO format
- Basic 3D modeling concepts

### Steps
1. Create a simple robot model using URDF or XACRO
2. Add appropriate collision and visual elements
3. Include joint definitions for robot articulation
4. Spawn the custom model in Gazebo
5. Verify physics properties and visual appearance
6. Test basic movement in simulation

### Resources Required
- Text editor for URDF/XACRO files
- Course-provided Gazebo plugins
- Custom robot model files

### Validation Criteria
- Robot model loads correctly in Gazebo
- Physics properties behave as expected
- Visual elements display properly
- Robot articulation works correctly

---

## Lab Exercise 3: Sensor Integration in Simulation

### Objective
Students will add and configure various sensors to their robot model in simulation.

### Prerequisites
- Lab Exercise 2 completed
- Understanding of robot sensors (RGB-D, LiDAR, IMU)
- ROS 2 topic communication

### Steps
1. Add RGB-D camera sensor to the robot model
2. Configure LiDAR sensor for distance measurements
3. Integrate IMU sensor for orientation data
4. Publish sensor data to ROS 2 topics
5. Visualize sensor data using ROS 2 tools
6. Test sensor behavior in different environments

### Resources Required
- Custom robot model from Lab 2
- Gazebo sensor plugins
- ROS 2 visualization tools (RViz2)

### Validation Criteria
- All sensors publish data to appropriate topics
- Sensor data is accurate and noise-free
- Data formats match ROS 2 standard message types
- Visualization tools display sensor data correctly

---

## Lab Exercise 4: Physics Configuration and Parameter Tuning

### Objective
Students will configure physics parameters to achieve realistic simulation behavior.

### Prerequisites
- Lab Exercise 3 completed
- Understanding of physics concepts (friction, mass, damping)
- Basic mathematics knowledge

### Steps
1. Analyze the physics properties of your robot model
2. Configure material properties for realistic interaction
3. Adjust mass and inertia parameters for stability
4. Tune friction and damping coefficients
5. Test performance in various scenarios
6. Compare simulation to expected real-world behavior

### Resources Required
- Robot model with sensors from Lab 3
- Physics analysis tools
- Reference documentation for physics parameters

### Validation Criteria
- Robot model behaves stably in simulation
- Physics parameters result in realistic movement
- Robot doesn't exhibit unexpected behaviors
- Performance is stable across different scenarios

---

## Lab Exercise 5: Complex Environment Design

### Objective
Students will create a complex environment with multiple obstacles and scenarios.

### Prerequisites
- All previous lab exercises completed
- Understanding of Gazebo world format
- Basic 3D modeling concepts

### Steps
1. Design a complex indoor environment with furniture
2. Add dynamic obstacles that move during simulation
3. Create specific scenarios for testing robot capabilities
4. Implement environment-specific physics configurations
5. Test robot navigation in the complex environment
6. Document performance metrics and observations

### Resources Required
- Gazebo world editor
- 3D models for environment elements
- Robot model with all sensors

### Validation Criteria
- Environment includes multiple complex elements
- Dynamic obstacles behave correctly
- Scenarios adequately test robot capabilities
- Robot performance is documented and analyzed

---

## Lab Exercise 6: Introduction to Isaac Sim (Optional)

### Objective
Students will explore NVIDIA Isaac Sim for high-fidelity sensor simulation.

### Prerequisites
- All Gazebo exercises completed
- Access to GPU with NVIDIA drivers
- Basic understanding of USD format

### Steps
1. Install NVIDIA Isaac Sim
2. Load a robot model into Isaac Sim
3. Configure high-fidelity sensors
4. Compare sensor output with Gazebo
5. Analyze differences in simulation quality
6. Document advantages and disadvantages

### Resources Required
- NVIDIA GPU with compatible drivers
- Isaac Sim installation
- Robot model files in USD or compatible format

### Validation Criteria
- Successfully load robot in Isaac Sim
- Sensors provide high-fidelity data
- Comparison with Gazebo shows differences
- Advantages of Isaac Sim are identified