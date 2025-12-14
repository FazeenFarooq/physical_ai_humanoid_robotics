# Unity Simulation Environment Templates for Physical AI Course

This document outlines the structure and components for Unity simulation environments used in the Physical AI & Humanoid Robotics course.

## Directory Structure
```
unity_envs/
├── scenes/
│   ├── basic_room.unity
│   ├── indoor_lab.unity
│   └── outdoor_area.unity
├── prefabs/
│   ├── robot_humanoid.prefab
│   ├── sensor_package.prefab
│   └── environment_objects.prefab
├── materials/
│   ├── robot_material.mat
│   └── environment_materials.mat
├── scripts/
│   ├── RobotController.cs
│   ├── SensorInterface.cs
│   └── ROSConnector.cs
├── configs/
│   ├── sim_config.json
│   └── robot_params.json
└── assets/
    └── 3D_models/
```

## Configuration Template: sim_config.json

```json
{
  "simulationSettings": {
    "timeStep": 0.01,
    "gravity": {
      "x": 0,
      "y": -9.81,
      "z": 0
    },
    "solverIterations": 6,
    "solverVelocityIterations": 1
  },
  "robotSettings": {
    "modelName": "basic_humanoid",
    "baseController": "PID",
    "controlFrequency": 100,
    "sensorFrequency": 30
  },
  "sensorConfig": {
    "camera": {
      "enabled": true,
      "resolution": {
        "width": 640,
        "height": 480
      },
      "fov": 60,
      "range": 10
    },
    "lidar": {
      "enabled": true,
      "rays": 360,
      "range": 10,
      "rotationSpeed": 10
    },
    "imu": {
      "enabled": true,
      "noise": 0.01
    }
  },
  "environmentSettings": {
    "enablePhysics": true,
    "contactPairsMode": "EnableAll",
    "sleepThreshold": 0.005
  }
}
```

## Unity Script Template: RobotController.cs

```csharp
using UnityEngine;

public class RobotController : MonoBehaviour
{
    // Robot configuration
    [Header("Robot Configuration")]
    public float maxVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;
    
    // Components
    private Rigidbody rb;
    private ArticulationBody[] joints;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        joints = GetComponentsInChildren<ArticulationBody>();
    }
    
    void Update()
    {
        // Handle control commands from ROS (not implemented in this template)
        // This would receive commands via ROS# or similar bridge
    }
    
    public void SetJointPositions(float[] positions)
    {
        if (joints.Length == positions.Length)
        {
            for (int i = 0; i < joints.Length; i++)
            {
                ArticulationDrive drive = joints[i].xDrive;
                drive.target = positions[i];
                joints[i].xDrive = drive;
            }
        }
    }
    
    public float[] GetJointPositions()
    {
        float[] positions = new float[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            positions[i] = joints[i].xDrive.target;
        }
        return positions;
    }
}
```

## ROS Integration Template

Unity can connect to ROS using the Unity Robotics Hub. The ROS connector would handle:

- Publishing sensor data to ROS topics
- Subscribing to control commands from ROS
- Maintaining time synchronization
- Converting Unity coordinates to ROS coordinates

## Physics Parameters

Unity uses PhysX for physics simulation. The following parameters can be configured:

- Material properties (friction, bounciness)
- Collision detection settings
- Solver parameters for accuracy vs. performance

## Export Configuration

For use with ROS bridge, ensure:

- Correct coordinate system (ROS uses right-handed system)
- Appropriate mesh simplification for performance
- Proper scaling between Unity units and real-world measurements
```

This serves as a template for Unity simulation environments that would be used in the course.