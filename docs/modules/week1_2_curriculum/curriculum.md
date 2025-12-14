# Week 1-2 Curriculum: ROS 2 and Digital Twin Fundamentals

This document outlines the curriculum for the first two weeks of the Physical AI & Humanoid Robotics course, focusing on ROS 2 and Digital Twin fundamentals.

## Week 1: Course Introduction & ROS 2 Fundamentals

### Learning Objectives
- Understand Physical AI and Embodied Intelligence concepts
- Set up development environment with ROS 2 Humble
- Master ROS 2 communication patterns (topics, services, actions)
- Create and run basic ROS 2 nodes for robotic communication

### Theory Topics
1. Introduction to Physical AI and Embodied Intelligence
   - Definition of Physical AI systems
   - Importance of sensorimotor coupling
   - Transition from digital-only to physically grounded intelligence

2. ROS 2 Fundamentals
   - Architecture overview (DDS-based communication)
   - Nodes, topics, services, and actions
   - Launch files and parameters
   - Package management and workspace setup

3. Robot Operating System Concepts
   - Distributed computing for robotics
   - Middleware for robot applications
   - Lifecycle nodes and managed processes

### Hands-on Labs
1. Environment Setup Lab
   - Install ROS 2 Humble Hawksbill
   - Configure development environment
   - Create first ROS 2 workspace
   - Verify installation with basic commands

2. Basic Communication Lab
   - Create publisher and subscriber nodes
   - Implement custom message types
   - Test communication between nodes
   - Use ROS 2 tools for introspection

3. Services and Actions Lab
   - Implement service server and client
   - Create action server and client
   - Compare request-response vs. goal-oriented patterns
   - Handle failures and timeouts

### Deliverables
- Working ROS 2 environment with publisher/subscriber nodes
- Custom message definitions and usage
- Service and action implementations
- Documentation of environment setup process

### Toolchain Used
- ROS 2 Humble Hawksbill
- Python 3.10
- Linux tools (bash, apt, etc.)
- rqt tools for visualization
- RViz2 for 3D visualization

## Week 2: Digital Twin Fundamentals

### Learning Objectives
- Create and configure simulation environments
- Understand physics simulation principles
- Integrate robot models with simulation environments
- Master Gazebo Garden for physics simulation

### Theory Topics
1. Digital Twin Concepts
   - Definition and purpose of digital twins in robotics
   - Simulation vs. reality considerations
   - Benefits of simulation-first methodology

2. Physics Simulation
   - Collision detection and response
   - Mass, friction, and damping parameters
   - Real-time physics constraints

3. Sensor Modeling
   - Types of simulated sensors (RGB-D, LiDAR, IMU)
   - Noise modeling and sensor accuracy
   - Integration with ROS 2 topics

### Hands-on Labs
1. Gazebo Environment Lab
   - Install Gazebo Garden
   - Create custom world with obstacles
   - Spawn robot model in simulation
   - Control robot in simulation environment

2. Robot Model Integration Lab
   - Create URDF model of simple robot
   - Add collision and visual elements
   - Integrate sensors into robot model
   - Test robot in simulation environment

3. Sensor Configuration Lab
   - Configure RGB-D camera in simulation
   - Set up LiDAR sensor for distance measurements
   - Integrate IMU sensor for orientation
   - Verify sensor data publication to ROS 2

### Deliverables
- Custom Gazebo world with robot model
- Robot model with integrated sensors
- Working sensor data publication to ROS 2
- Documentation of simulation setup process

### Toolchain Used
- Gazebo Garden (fortified)
- URDF/XACRO models
- Gazebo plugins
- ROS 2 sensor message types
- RViz2 for visualization

## Assessment Criteria

### Week 1 Assessment
- Students can create and run basic ROS 2 nodes
- Communication between nodes works correctly
- Custom messages are properly defined and used
- Students understand the difference between services and actions

### Week 2 Assessment
- Students can create custom Gazebo environments
- Robot model integrates properly with simulation
- Sensors publish data to correct ROS 2 topics
- Simulation behaves according to physics principles

## Prerequisites for Next Weeks
- ROS 2 workspace properly configured
- Gazebo simulation environment working
- Basic understanding of robot communication
- Ability to integrate sensors with robot models

## Troubleshooting Common Issues
1. ROS 2 Installation Issues
   - Domain ID conflicts in multi-robot environments
   - Library path configuration
   - Permission issues with device access

2. Simulation Problems
   - Robot model spawning failures
   - Physics instability
   - Sensor data not publishing correctly

## Resources for Further Learning
- ROS 2 Documentation for Humble Hawksbill
- Gazebo Garden Tutorials
- Physical AI research papers
- Course-specific code examples and templates

This curriculum provides the foundation for the remainder of the course, ensuring students have the necessary skills in ROS 2 communication and simulation environments before advancing to more complex topics.