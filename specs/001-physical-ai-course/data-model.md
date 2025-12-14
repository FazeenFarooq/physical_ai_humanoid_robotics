# Data Model: Physical AI & Humanoid Robotics Course

**Feature**: 001-physical-ai-course
**Date**: 2025-12-14

## Overview

This document defines the data models used in the Physical AI & Humanoid Robotics course systems. These models represent the core entities and their relationships that students will work with throughout the course, from simulation to physical robot implementation.

## Core Entities

### Student
- **ID**: Unique identifier for each student
- **Name**: Full name of the student
- **Email**: Contact email address
- **EnrollmentDate**: Date of course enrollment
- **Prerequisites**: List of verified prerequisites (Python, ML, Linux, etc.)
- **CurrentModule**: Module currently being worked on
- **Progress**: Percentage completion of course
- **LabSubmissions**: List of completed lab assignments
- **CapstoneStatus**: Current status of capstone project

### Module
- **ID**: Unique identifier for each module
- **Name**: Name of the module (e.g., "ROS 2 Robotic Nervous System")
- **Description**: Summary of module content and objectives
- **Duration**: Estimated time to complete (in weeks)
- **Prerequisites**: Skills or modules required before starting
- **Objectives**: List of learning objectives
- **TheoryTopics**: List of theoretical concepts covered
- **LabExercises**: Collection of hands-on lab exercises
- **Deliverables**: Required outputs from the module
- **Toolchain**: Software and hardware tools used

### LabExercise
- **ID**: Unique identifier for the lab exercise
- **Name**: Name of the lab exercise
- **Description**: Detailed description of the exercise
- **Objectives**: Learning objectives for this lab
- **Steps**: Sequential steps to complete the lab
- **ValidationCriteria**: How to verify successful completion
- **Resources**: Required files, models, or environments
- **Difficulty**: Level of complexity (1-5)
- **EstimatedTime**: Time required to complete (in hours)
- **RelatedModule**: Module this lab belongs to

### RobotEnvironment
- **ID**: Unique identifier for the environment
- **Name**: Name of the environment (e.g., "Home Environment", "Lab Space")
- **Description**: Description of the environment layout
- **Type**: Simulation (Gazebo, Isaac Sim, Unity) or Physical
- **Models**: List of robot models supported in this environment
- **Obstacles**: Static and dynamic objects in the environment
- **Sensors**: Sensor configurations available
- **PhysicsParameters**: Physics engine settings
- **Tasks**: List of tasks that can be performed in this environment

### RobotModel
- **ID**: Unique identifier for the robot model
- **Name**: Name of the robot (e.g., "Unitree H1", "TurtleBot 4")
- **Description**: Physical description and specifications
- **KinematicModel**: URDF/XACRO representation
- **Actuators**: Joint motors and their specifications
- **Sensors**: Sensor configurations (RGB-D, LiDAR, IMU, audio)
- **ComputationalResources**: Available processing power, memory
- **BatteryLife**: Operational time on full charge
- **Workspace**: Reachable area and volume

### TaskPlan
- **ID**: Unique identifier for the task
- **Name**: Name of the high-level task
- **Description**: Natural language description of the task
- **Steps**: Sequential steps to complete the task
- **Requirements**: Resources and conditions needed
- **Constraints**: Limitations or restrictions
- **SuccessCriteria**: How to determine task completion
- **FallbackBehaviors**: Actions to take if primary approach fails
- **EstimatedTime**: Expected time for completion

### PerceptionData
- **ID**: Unique identifier for the data collection
- **Timestamp**: When the data was captured
- **SensorType**: Type of sensor (RGB-D, LiDAR, IMU, Audio)
- **Data**: Raw sensor data
- **Environment**: Environment where data was collected
- **RobotState**: Robot's state when data was collected
- **Annotations**: Labels or annotations for the data
- **Source**: Simulation or physical robot
- **QualityScore**: Measure of data quality

### ActionCommand
- **ID**: Unique identifier for the command
- **Type**: Type of action (navigation, manipulation, interaction)
- **Parameters**: Specific parameters for the action
- **Priority**: Priority level of the action
- **Timestamp**: When command was issued
- **Status**: Queued, executing, completed, failed
- **Executor**: Component responsible for executing the command
- **Dependencies**: Other commands that must complete first

### CapstoneProject
- **ID**: Unique identifier for the capstone
- **StudentID**: Student who owns this capstone
- **Milestone1Status**: Status of Voice-to-Intent milestone
- **Milestone2Status**: Status of Perception & Mapping milestone
- **Milestone3Status**: Status of Navigation & Obstacle Avoidance milestone
- **Milestone4Status**: Status of Object Identification & Manipulation milestone
- **FinalDemoStatus**: Status of final demonstration
- **Components**: List of integrated system components
- **PerformanceMetrics**: Actual performance measurements
- **FailureAnalysis**: Analysis of failed attempts and lessons learned
- **FinalScore**: Overall evaluation score

### HardwareResource
- **ID**: Unique identifier for the hardware
- **Type**: Workstation, Jetson Orin, Robot Platform, etc.
- **Model**: Specific model and specifications
- **Location**: Physical location of the hardware
- **Status**: Available, Reserved, In Use, Maintenance, Faulty
- **Reservation**: Current reservation information
- **Owner**: Person responsible for maintenance
- **LastCalibration**: Date of last calibration
- **AvailabilitySchedule**: When the hardware is available

## Relationships

### Student-Module
- A Student enrolls in one or more Modules
- A Module is completed by many Students
- Relationship: Many-to-Many (with enrollment data)

### Module-LabExercise
- A Module contains many LabExercises
- A LabExercise belongs to one Module
- Relationship: One-to-Many

### Student-LabExercise
- A Student completes many LabExercises
- A LabExercise is completed by many Students
- Relationship: Many-to-Many (with completion data)

### RobotEnvironment-RobotModel
- A RobotEnvironment supports many RobotModels
- A RobotModel can be used in many RobotEnvironments
- Relationship: Many-to-Many

### TaskPlan-ActionCommand
- A TaskPlan consists of many ActionCommands
- An ActionCommand belongs to one TaskPlan
- Relationship: One-to-Many

### Student-CapstoneProject
- A Student has one CapstoneProject
- A CapstoneProject belongs to one Student
- Relationship: One-to-One

### Student-HardwareResource
- A Student reserves many HardwareResources
- A HardwareResource is reserved by many Students over time
- Relationship: Many-to-Many (with reservation data)

## State Transitions

### LabExercise State Transitions
- **Not Started** → **In Progress**: Student begins the lab
- **In Progress** → **Completed**: Student submits successful solution
- **In Progress** → **Failed**: Student's solution doesn't meet criteria
- **Failed** → **In Progress**: Student retries the lab

### CapstoneProject State Transitions
- **Not Started** → **Milestone 1**: Student begins Voice-to-Intent
- **Milestone 1** → **Milestone 2**: Voice-to-Intent complete
- **Milestone 2** → **Milestone 3**: Perception & Mapping complete
- **Milestone 3** → **Milestone 4**: Navigation & Avoidance complete
- **Milestone 4** → **Integration**: All milestones complete, integration begins
- **Integration** → **Demo Preparation**: System integrated, preparing for demo
- **Demo Preparation** → **Completed**: Final demonstration successful

### HardwareResource State Transitions
- **Available** → **Reserved**: Student reserves the hardware
- **Reserved** → **In Use**: Student begins using hardware
- **In Use** → **Available**: Student finishes with hardware
- **In Use** → **Maintenance**: Hardware needs maintenance
- **Maintenance** → **Available**: Maintenance completed
- **Any State** → **Faulty**: Hardware malfunctions
- **Faulty** → **Maintenance**: Faulty hardware is fixed

## Validation Rules

### Student Entity
- Name must be 2-50 characters
- Email must be valid format
- Prerequisites must satisfy course requirements
- CurrentModule must be valid module in course sequence

### Module Entity
- Duration must be 1-4 weeks for course context
- Objectives must be measurable and specific
- Deliverables must be verifiable

### LabExercise Entity
- Steps must be executable in sequence
- ValidationCriteria must be objective and measurable
- EstimatedTime must be between 1-20 hours

### RobotEnvironment Entity
- All sensor configurations must be compatible with available robot models
- Physics parameters must be valid for simulation engine

### TaskPlan Entity
- Steps must form a logical sequence
- SuccessCriteria must be objectively measurable
- FallbackBehaviors must not create infinite loops

### CapstoneProject Entity
- Each milestone must build on previous components
- PerformanceMetrics must be consistent with course requirements
- All milestones must be completed before final demo