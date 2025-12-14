# ROS 2 Fundamentals Lab Exercises

This document outlines the lab exercises for the ROS 2 Robotic Nervous System module of the Physical AI & Humanoid Robotics course.

## Lab Exercise 1: ROS 2 Environment Setup

### Objective
Students will set up a complete ROS 2 development environment and verify basic functionality.

### Prerequisites
- Basic understanding of Linux command line
- Python programming knowledge
- Familiarity with package managers

### Steps
1. Install ROS 2 Humble Hawksbill on your development machine
2. Set up the ROS 2 environment variables
3. Verify installation by running basic ROS 2 commands
4. Create your first ROS 2 workspace

### Resources Required
- Linux development machine
- Internet connection for package downloads
- Course-provided setup scripts

### Validation Criteria
- Successfully install ROS 2 Humble
- Source the ROS 2 environment without errors
- Run `ros2 topic list` without errors
- Create and build a basic workspace

---

## Lab Exercise 2: Publisher/Subscriber Communication

### Objective
Students will create and run a simple publisher and subscriber to understand ROS 2 communication patterns.

### Prerequisites
- ROS 2 environment setup
- Basic Python knowledge

### Steps
1. Create a new ROS 2 package for the lab
2. Implement a publisher node that sends messages at 1Hz
3. Implement a subscriber node that receives and logs messages
4. Test the communication between nodes
5. Modify the message content and observe changes

### Resources Required
- Completed lab exercise 1
- Basic Python IDE

### Validation Criteria
- Publisher sends messages at consistent 1Hz rate
- Subscriber receives messages with minimal delay
- Students can modify message content successfully
- No errors in ROS 2 communication

---

## Lab Exercise 3: Services and Actions

### Objective
Students will implement services and actions to understand request-response patterns and goal-oriented behaviors.

### Prerequisites
- Lab exercises 1 and 2 completed
- Understanding of ROS 2 topics

### Steps
1. Create a service server that performs simple calculations
2. Create a service client that makes requests to the server
3. Implement an action server for a simple navigation goal
4. Create an action client that sends goals and receives feedback
5. Test robustness with multiple clients

### Resources Required
- ROS 2 workspace with publisher/subscriber code
- Basic understanding of asynchronous programming

### Validation Criteria
- Service server correctly processes requests and sends responses
- Action server provides feedback and result appropriately
- Client handles service responses and action feedback
- Error handling works for failed requests/goals

---

## Lab Exercise 4: Launch Files and Parameter Management

### Objective
Students will learn to use launch files to start multiple nodes simultaneously and manage parameters.

### Prerequisites
- All previous lab exercises completed
- Understanding of ROS 2 nodes and communication

### Steps
1. Create a launch file to start publisher and subscriber nodes
2. Add parameters to nodes and configure them via launch file
3. Use conditional launching based on parameters
4. Handle node lifecycles using launch file directives
5. Create launch arguments to customize behavior

### Resources Required
- Code from previous lab exercises
- Understanding of YAML format

### Validation Criteria
- Launch file successfully starts all nodes
- Parameters are correctly passed to nodes
- Launch arguments work as expected
- Nodes behave differently based on parameters

---

## Lab Exercise 5: Debugging and Tooling

### Objective
Students will use ROS 2 tools to debug and visualize their robotic systems.

### Prerequisites
- All previous lab exercises completed
- Understanding of ROS 2 communication patterns

### Steps
1. Use `rqt` tools to visualize system state
2. Monitor topics using `ros2 topic` commands
3. Use introspection to examine message contents
4. Profile system performance using ROS 2 tools
5. Debug common communication issues

### Resources Required
- Working ROS 2 nodes with various topics and services
- Access to debugging tools (rqt, rviz2, etc.)

### Validation Criteria
- Students can visualize system state using appropriate tools
- Students can monitor and analyze system communications
- Students can identify and fix common issues
- Performance profiling is performed and analyzed