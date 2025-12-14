# ROS 2 Topics, Services, and Actions Lab Exercises

This document outlines the lab exercises for Module 1 of the Physical AI & Humanoid Robotics course, focusing on ROS 2 communication patterns: topics, services, and actions.

## Lab Exercise 1: Advanced Publisher/Subscriber Patterns

### Objective
Students will implement more complex publisher/subscriber communication patterns beyond basic message passing.

### Prerequisites
- Basic ROS 2 environment setup
- Understanding of basic publisher/subscriber concepts
- Completed Week 1-2 curriculum materials

### Learning Outcomes
- Implement publishers with different QoS settings
- Handle message synchronization between multiple topics
- Create custom message types for specialized data

### Steps
1. Create a custom message type for sensor fusion data that combines information from multiple sensors
2. Implement a publisher that sends sensor data at different frequencies
3. Create subscribers that handle messages with different QoS policies (reliable vs. best effort)
4. Implement message synchronization using ROS 2 message filters to combine data from multiple sensors
5. Test the system with simulated sensor data to ensure proper synchronization

### Resources Required
- ROS 2 Humble environment
- Python 3.10+
- Basic understanding of message definition files (.msg)

### Validation Criteria
- Custom message type compiles without errors
- Publishers and subscribers communicate correctly
- Message synchronization works properly
- QoS policies are appropriately applied

### Estimated Time
4-6 hours

---

## Lab Exercise 2: Service-Based Robot Control

### Objective
Students will create and use services for robot control commands in a request-response pattern.

### Prerequisites
- Lab Exercise 1 completed
- Understanding of ROS 2 services
- Basic knowledge of robot kinematics

### Learning Outcomes
- Implement a service server for robot joint control
- Create a service client that sends control commands
- Handle service request validation and error responses
- Implement timeout and retry mechanisms

### Steps
1. Design a service interface for setting robot joint positions with validation
2. Implement a service server that accepts joint position requests and validates them
3. Create a service client that sends joint position commands and handles responses
4. Add safety checks to ensure requested positions are within joint limits
5. Implement timeout handling and retry logic for failed service calls
6. Test the service with various joint position requests

### Resources Required
- Robot model with joint information
- Understanding of robot joint limits
- Service implementation knowledge

### Validation Criteria
- Service server validates requests properly
- Client handles responses and errors appropriately
- Safety checks prevent invalid joint positions
- Timeout and retry mechanisms work correctly

### Estimated Time
5-7 hours

---

## Lab Exercise 3: Action-Based Navigation System

### Objective
Students will implement a goal-oriented navigation system using ROS 2 actions.

### Prerequisites
- Lab Exercises 1 and 2 completed
- Understanding of ROS 2 actions
- Basic understanding of navigation concepts

### Learning Outcomes
- Implement an action server for navigation goals
- Create an action client for sending navigation requests
- Handle goal feedback and cancellation
- Implement goal preemption for dynamic replanning

### Steps
1. Design an action interface for navigation to 2D poses with feedback
2. Implement an action server that simulates navigation to specified coordinates
3. Create an action client that sends navigation goals and monitors progress
4. Implement feedback publishing to update navigation progress
5. Add goal cancellation and preemption capabilities
6. Test with various navigation scenarios and goal changes

### Resources Required
- Action implementation knowledge
- Navigation simulation environment
- Coordinate system understanding

### Validation Criteria
- Action server handles goals correctly
- Client properly sends goals and receives feedback
- Cancellation and preemption work as expected
- Navigation simulation behaves realistically

### Estimated Time
6-8 hours

---

## Lab Exercise 4: Multi-Node Communication Architecture

### Objective
Students will design and implement a multi-node system that uses all three communication patterns effectively.

### Prerequisites
- All previous lab exercises completed
- Understanding of all ROS 2 communication patterns
- Basic system architecture concepts

### Learning Outcomes
- Design a distributed system architecture
- Appropriately select communication patterns for different needs
- Implement system monitoring and error handling
- Create a cohesive multi-node application

### Steps
1. Design a system architecture for a mobile manipulator (robot with navigation and manipulation capabilities)
2. Identify which components should use topics, services, or actions
3. Implement the navigation subsystem using actions for goal-oriented tasks
4. Implement the manipulation subsystem using services for request-response interactions
5. Implement sensor data distribution using topics
6. Create a coordination node that integrates all subsystems
7. Add system monitoring using the ROS 2 debugging tools
8. Test the complete system with integrated scenarios

### Resources Required
- All previous implementations
- Robot model with navigation and manipulation capabilities
- ROS 2 debugging tools
- System design experience

### Validation Criteria
- Appropriate communication patterns used for each requirement
- System components work together cohesively
- Error handling and monitoring are effective
- Overall system performs integrated tasks successfully

### Estimated Time
8-10 hours

---

## Lab Exercise 5: Performance Optimization and Diagnostics

### Objective
Students will optimize their ROS 2 system for performance and use diagnostic tools to identify bottlenecks.

### Prerequisites
- Lab Exercise 4 completed
- Understanding of ROS 2 performance considerations
- Experience with the ROS 2 debugger tools

### Learning Outcomes
- Identify performance bottlenecks in ROS 2 systems
- Apply optimization techniques to improve performance
- Use diagnostic tools for system analysis
- Validate performance improvements

### Steps
1. Profile the multi-node system developed in Lab Exercise 4 for performance
2. Identify bottlenecks using the ROS 2 debugger tools
3. Optimize message publishing rates for real-time requirements
4. Adjust QoS settings to improve communication performance
5. Implement performance monitoring in the system
6. Document performance improvements and validation results

### Resources Required
- Multi-node system from Lab Exercise 4
- ROS 2 debugger tools
- Performance profiling tools
- Understanding of QoS policies

### Validation Criteria
- Performance bottlenecks are identified and addressed
- System performance improves after optimization
- Diagnostic tools provide useful information
- Performance metrics are validated

### Estimated Time
6-8 hours

---

## Lab Exercise 6: Integration with Simulation (Optional Extension)

### Objective
Students will integrate their ROS 2 communication system with a Gazebo simulation environment.

### Prerequisites
- All previous lab exercises completed
- Basic Gazebo simulation knowledge
- Completed Digital Twin fundamentals

### Learning Outcomes
- Integrate ROS 2 nodes with Gazebo simulation
- Handle simulation-specific communication requirements
- Validate system behavior in simulated environment

### Steps
1. Connect the multi-node system to a Gazebo simulation environment
2. Map ROS 2 topics to Gazebo plugins for robot control
3. Validate that communication patterns work correctly in simulation
4. Test system behavior with simulated sensors and actuators
5. Compare performance between real and simulated environments

### Resources Required
- Gazebo simulation environment
- Robot model for simulation
- Gazebo ROS plugins knowledge

### Validation Criteria
- System communicates correctly with Gazebo simulation
- Performance is acceptable in simulation environment
- Behavior matches expectations from real hardware

### Estimated Time
5-7 hours

---

## Assessment Rubric

Each lab exercise will be assessed based on the following criteria:

### Implementation (40%)
- Correct implementation of ROS 2 communication patterns
- Proper use of ROS 2 APIs and best practices
- Code quality and documentation

### Functionality (30%)
- System works as expected without errors
- All features function correctly
- Error handling is implemented appropriately

### Design (20%)
- Appropriate selection of communication pattern (topic/service/action)
- Clean system architecture
- Effective use of ROS 2 features

### Validation (10%)
- Proper testing and validation of functionality
- Performance analysis where applicable
- Documentation of results and findings

## Submission Requirements

For each lab exercise, students must submit:

1. **Source Code**: All implemented ROS 2 nodes and related files
2. **Documentation**: Brief explanation of design decisions and challenges overcome
3. **Results**: Summary of validation results and performance metrics
4. **Reflection**: What was learned and how it connects to Physical AI principles

## Common Challenges and Solutions

1. **Node Discovery Issues**: Check network configuration and ROS_DOMAIN_ID
2. **Message Synchronization**: Use appropriate QoS settings and message filters
3. **Performance Bottlenecks**: Profile the system and optimize critical paths
4. **Integration Complexity**: Start with simple systems and incrementally add complexity

This lab exercise series provides hands-on experience with all major ROS 2 communication patterns while building toward a comprehensive robotic system as required by the Physical AI & Humanoid Robotics course constitution.