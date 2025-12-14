# API Contracts: Physical AI & Humanoid Robotics Course

**Feature**: 001-physical-ai-course
**Date**: 2025-12-14

## Overview

This document defines the API contracts for the various systems and interfaces used in the Physical AI & Humanoid Robotics course. These contracts specify how different components communicate and interact with each other.

## ROS 2 Message Definitions

### Perception Services

#### ObjectDetection Service
- **Request**:
  - `image`: sensor_msgs/Image - Input RGB image for object detection
  - `confidence_threshold`: float64 - Minimum confidence for detection
- **Response**:
  - `objects`: array of ObjectInfo - Detected objects with positions and classes
  - `success`: bool - Whether the detection was successful
  - `message`: string - Additional information about the detection

#### ObjectInfo Message
- `class_name`: string - Class of the detected object
- `confidence`: float64 - Detection confidence (0.0 to 1.0)
- `bbox`: BoundingBox - 2D bounding box coordinates
- `position_3d`: geometry_msgs/Point - 3D position in world coordinates

#### BoundingBox Message
- `x_min`: float64 - Minimum X coordinate
- `y_min`: float64 - Minimum Y coordinate
- `x_max`: float64 - Maximum X coordinate
- `y_max`: float64 - Maximum Y coordinate

### Navigation Services

#### PathPlanning Service
- **Request**:
  - `start_pose`: geometry_msgs/Pose - Starting position
  - `goal_pose`: geometry_msgs/Pose - Target position
  - `map_name`: string - Name of the map to use for planning
- **Response**:
  - `path`: nav_msgs/Path - Planned path as sequence of poses
  - `success`: bool - Whether path planning was successful
  - `message`: string - Additional information

#### Navigation Action
- **Goal**:
  - `target_pose`: geometry_msgs/Pose - Destination for navigation
  - `planner_id`: string - ID of the planner to use
- **Result**:
  - `path`: nav_msgs/Path - Executed path
  - `distance_traveled`: float64 - Total distance traveled
  - `success`: bool - Whether navigation was successful
- **Feedback**:
  - `distance_remaining`: float64 - Distance to goal
  - `current_speed`: float64 - Current navigation speed
  - `state`: string - Current navigation state

### Manipulation Services

#### GraspPlanning Service
- **Request**:
  - `object_info`: ObjectInfo - Information about object to grasp
  - `robot_state`: sensor_msgs/JointState - Current robot joint positions
- **Response**:
  - `grasp_pose`: geometry_msgs/Pose - Recommended grasp pose
  - `approach_direction`: geometry_msgs/Vector3 - Approach direction
  - `success`: bool - Whether grasp planning was successful
  - `failure_reason`: string - Reason if planning failed

#### Manipulation Action
- **Goal**:
  - `target_object`: ObjectInfo - Object to manipulate
  - `manipulation_type`: string - Type of manipulation (grasp, place, move)
  - `target_pose`: geometry_msgs/Pose - Target pose for manipulation
- **Result**:
  - `manipulation_success`: bool - Whether manipulation was successful
  - `final_pose`: geometry_msgs/Pose - Final pose of object
  - `error_message`: string - Error details if applicable
- **Feedback**:
  - `current_phase`: string - Current manipulation phase
  - `grasp_success`: bool - Whether grasp was successful
  - `execution_progress`: float64 - Progress percentage

### Conversation Services

#### SpeechRecognition Service
- **Request**:
  - `audio_data`: audio_common_msgs/AudioData - Audio input
  - `language`: string - Language of speech (default: "en-US")
- **Response**:
  - `transcript`: string - Transcribed text
  - `confidence`: float64 - Confidence of transcription (0.0 to 1.0)
  - `success`: bool - Whether recognition was successful

#### NaturalLanguageUnderstanding Service
- **Request**:
  - `input_text`: string - Text to process
  - `context`: string - Context information for understanding
- **Response**:
  - `intent`: string - Recognized intent
  - `entities`: array of Entity - Extracted entities
  - `confidence`: float64 - Confidence of understanding (0.0 to 1.0)
  - `success`: bool - Whether understanding was successful

#### Entity Message
- `type`: string - Type of entity (object, location, action)
- `value`: string - Entity value
- `confidence`: float64 - Confidence of entity recognition

## Topic Interfaces

### Sensor Data Topics
- `/camera/rgb/image_raw` - Raw RGB camera images (sensor_msgs/Image)
- `/camera/depth/image_raw` - Raw depth images (sensor_msgs/Image)
- `/lidar/points` - LiDAR point cloud data (sensor_msgs/PointCloud2)
- `/imu/data` - IMU sensor readings (sensor_msgs/Imu)
- `/robot/joint_states` - Joint position and velocity (sensor_msgs/JointState)

### Control Topics
- `/cmd_vel` - Velocity commands for base movement (geometry_msgs/Twist)
- `/joint_group_position_controller/commands` - Joint position commands (std_msgs/Float64MultiArray)
- `/gripper_controller/commands` - Gripper commands (std_msgs/Float64)

### Status Topics
- `/robot/odometry` - Robot odometry (nav_msgs/Odometry)
- `/robot/battery_status` - Battery level and status (sensor_msgs/BatteryState)
- `/system/status` - Overall system status (std_msgs/String)

## Service Interface for Hardware Management

### HardwareResource Service
- **Request**:
  - `resource_id`: string - ID of the hardware resource
  - `action`: string - Requested action (reserve, release, status)
  - `duration`: int32 - Duration for reservation (if applicable)
  - `user_id`: string - ID of the user making request
- **Response**:
  - `success`: bool - Whether the action was successful
  - `status`: string - Current status of the resource
  - `message`: string - Additional information
  - `reservation_id`: string - ID of the reservation (if applicable)

## Error Handling

### Standard Error Response
All services and actions follow this error response pattern:

- `success`: bool - False if there was an error
- `error_code`: string - Standardized error code (e.g., "INVALID_INPUT", "RESOURCE_UNAVAILABLE", "EXECUTION_FAILED")
- `error_message`: string - Human-readable error description
- `timestamp`: builtin_interfaces/Time - Time when error occurred

### Error Codes
- `INVALID_INPUT` - Request parameters were invalid
- `RESOURCE_UNAVAILABLE` - Required resource is not available
- `EXECUTION_FAILED` - Action execution failed for unexpected reason
- `TIMEOUT` - Request timed out waiting for response
- `PERMISSION_DENIED` - User lacks permission for requested action
- `HARDWARE_FAULT` - Associated hardware reported a fault