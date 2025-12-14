# NVIDIA Isaac Sim Configuration for Physical AI Course

This directory contains configuration files and assets for NVIDIA Isaac Sim used in the Physical AI & Humanoid Robotics course.

## Isaac Sim Assets
- `humanoid_robot.usd`: USD representation of the humanoid robot model
- `sensor_config.usd`: Sensor configuration for Isaac Sim (RGB-D, LiDAR, IMU)
- `environments/`: Directory containing various simulation environments

## Physics Configuration
- `physics_materials.usda`: Physics materials for realistic simulation
- `contact_solvers.usda`: Contact solver configurations for accurate physics

## Simulation Scenarios
- `navigation_task.usd`: Navigation task scenario
- `manipulation_task.usd`: Manipulation task scenario
- `conversation_scenario.usd`: Human-robot interaction scenario

## Deployment Scripts
- `deploy_to_jetson.py`: Scripts to transfer trained models to Jetson Orin
- `sim_to_real_transfer.py`: Tools for sim-to-real transfer optimization