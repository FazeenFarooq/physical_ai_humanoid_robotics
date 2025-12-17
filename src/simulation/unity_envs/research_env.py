"""
Research-focused Unity Environment for Physical AI Experiments

This module provides interfaces and templates for creating advanced simulation 
environments in Unity specifically designed for conducting reproducible research 
experiments in the Physical AI & Humanoid Robotics course.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class UnitySceneConfig:
    """Configuration for a Unity research scene"""
    scene_name: str
    description: str
    physics_settings: Dict[str, Any]
    robot_prefabs: List[str]
    environment_objects: List[Dict[str, Any]]
    sensors: List[Dict[str, Any]]
    research_tasks: List[Dict[str, Any]]
    safety_boundaries: Dict[str, Any]


class UnityResearchEnvironment:
    """
    A class to manage research-focused Unity simulation environments.
    This class doesn't directly interface with Unity but provides 
    configuration and management for Unity scenes used in research.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Unity research environment.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.scenes = {}
        self.experiment_data = []
        
        self._setup_environment()
    
    def _load_config(self, config_path: str) -> UnitySceneConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            UnitySceneConfig object
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return UnitySceneConfig(
                scene_name=data['scene_name'],
                description=data['description'],
                physics_settings=data['physics_settings'],
                robot_prefabs=data['robot_prefabs'],
                environment_objects=data['environment_objects'],
                sensors=data['sensors'],
                research_tasks=data['research_tasks'],
                safety_boundaries=data['safety_boundaries']
            )
    
    def _default_config(self) -> UnitySceneConfig:
        """
        Create a default configuration for research environment.
        
        Returns:
            UnitySceneConfig with default values
        """
        return UnitySceneConfig(
            scene_name="default_research_scene",
            description="Default Unity research environment for Physical AI experiments",
            physics_settings={
                "gravity": -9.81,
                "solver_type": "SequentialImpulses",
                "solver_iteration_count": 6,
                "sleep_threshold": 0.005,
                "default_contact_offset": 0.01
            },
            robot_prefabs=[
                "TurtleBot4",
                "UnitreeH1",
                "CustomHumanoid"
            ],
            environment_objects=[
                {
                    "name": "table",
                    "type": "Static",
                    "dimensions": {"x": 1.2, "y": 0.8, "z": 0.75},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "material_friction": 0.5
                },
                {
                    "name": "obstacle_cylinder",
                    "type": "Dynamic",
                    "shape": "Cylinder",
                    "radius": 0.1,
                    "height": 0.5,
                    "position": {"x": 0.5, "y": 0.0, "z": 0.1},
                    "mass": 1.0,
                    "material_friction": 0.2
                }
            ],
            sensors=[
                {
                    "type": "RGB_Camera",
                    "resolution": {"width": 640, "height": 480},
                    "fov": 60,
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "rotation": {"x": -10, "y": 0, "z": 0}
                },
                {
                    "type": "Depth_Camera", 
                    "resolution": {"width": 640, "height": 480},
                    "fov": 60,
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "rotation": {"x": -10, "y": 0, "z": 0}
                },
                {
                    "type": "IMU",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.5},
                    "noise_std_dev": 0.01
                }
            ],
            research_tasks=[
                {
                    "name": "object_manipulation",
                    "description": "Manipulate objects of various shapes and weights",
                    "object_parameters": [
                        {"shape": "cube", "size": 0.1, "weight": 0.5},
                        {"shape": "sphere", "size": 0.05, "weight": 0.2}
                    ],
                    "metrics": ["success_rate", "manipulation_time", "energy_consumption"]
                },
                {
                    "name": "navigation", 
                    "description": "Navigate through obstacle courses",
                    "obstacle_parameters": [
                        {"type": "static", "complexity": "low"},
                        {"type": "dynamic", "complexity": "medium"}
                    ],
                    "metrics": ["path_efficiency", "collision_count", "completion_time"]
                }
            ],
            safety_boundaries={
                "workspace_volume": {"min": {"x": -2.0, "y": -2.0, "z": 0.0}, 
                                   "max": {"x": 2.0, "y": 2.0, "z": 2.0}},
                "robot_speed_limits": {"linear": 1.0, "angular": 1.5},
                "emergency_stop_conditions": ["collision", "boundary_violation", "timeout"]
            }
        )
    
    def _setup_environment(self):
        """
        Set up the environment based on configuration.
        """
        print(f"Setting up Unity research environment: {self.config.scene_name}")
        print(f"Description: {self.config.description}")
        
        # Create directory structure for Unity project
        self._create_directories()
        
        # Generate scene files
        self._generate_scene_files()
        
    def _create_directories(self):
        """
        Create necessary directories for Unity project.
        """
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        unity_project_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard Unity folder structure
        (unity_project_path / "Assets").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Scenes").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Prefabs").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Scripts").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Materials").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Models").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "Textures").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "PhysicsMaterials").mkdir(exist_ok=True)
        (unity_project_path / "Assets" / "ResearchData").mkdir(exist_ok=True)
        
        print(f"Created Unity project structure at: {unity_project_path}")
    
    def _generate_scene_files(self):
        """
        Generate Unity scene files based on configuration.
        """
        # Generate a Unity scene file (simplified representation)
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        scene_path = unity_project_path / "Assets" / "Scenes" / f"{self.config.scene_name}.unity"
        
        # Create a simplified Unity scene file as a text file
        scene_content = self._create_unity_scene_content()
        
        with open(scene_path, 'w') as f:
            f.write(scene_content)
        
        print(f"Generated Unity scene file: {scene_path}")
        
        # Generate configuration files
        config_path = unity_project_path / "Assets" / "ResearchData" / "scene_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "scene_name": self.config.scene_name,
                "description": self.config.description,
                "physics_settings": self.config.physics_settings,
                "environment_objects": self.config.environment_objects,
                "sensors": self.config.sensors,
                "research_tasks": self.config.research_tasks,
                "safety_boundaries": self.config.safety_boundaries
            }, f, indent=2)
        
        print(f"Generated configuration file: {config_path}")
    
    def _create_unity_scene_content(self) -> str:
        """
        Create content for a basic Unity scene file.
        
        Returns:
            String content of the Unity scene file
        """
        # This is a simplified representation of a Unity scene file
        scene_content = f"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!29 &1
OcclusionCullingSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 2
  m_OcclusionBakeSettings:
    smallestOccluder: 5
    smallestHole: 0.25
    backfaceThreshold: 100
  m_SceneGUID: 00000000000000000000000000000000
  m_OcclusionCullingData: {{fileID: 0}}
--- !u!104 &2
RenderSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 9
  m_Fog: 0
  m_FogColor: {{r: 0.5, g: 0.5, b: 0.5, a: 1}}
  m_FogMode: 3
  m_FogDensity: 0.01
  m_LinearFogStart: 0
  m_LinearFogEnd: 300
  m_AmbientSkyColor: {{r: 0.212, g: 0.227, b: 0.259, a: 1}}
  m_AmbientEquatorColor: {{r: 0.114, g: 0.125, b: 0.133, a: 1}}
  m_AmbientGroundColor: {{r: 0.047, g: 0.043, b: 0.035, a: 1}}
  m_AmbientIntensity: 1
  m_AmbientMode: 0
  m_SubtractiveShadowColor: {{r: 0.42, g: 0.478, b: 0.627, a: 1}}
  m_SkyboxMaterial: {{fileID: 10304, guid: 0000000000000000f000000000000000, type: 0}}
  m_HaloStrength: 0.5
  m_FlareStrength: 1
  m_FlareFadeSpeed: 3
  m_HaloTexture: {{fileID: 0}}
  m_SpotCookie: {{fileID: 10001, guid: 0000000000000000e000000000000000, type: 0}}
  m_DefaultReflectionMode: 0
  m_DefaultReflectionResolution: 128
  m_ReflectionBounces: 1
  m_ReflectionIntensity: 1
  m_CustomReflection: {{fileID: 0}}
  m_Sun: {{fileID: 0}}
  m_IndirectSpecularColor: {{r: 0, g: 0, b: 0, a: 1}}
  m_UseRadianceAmbientProbe: 0
--- !u!157 &3
LightmapSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 12
  m_GIWorkflowMode: 1
  m_GISettings:
    serializedVersion: 2
    m_BounceScale: 1
    m_IndirectOutputScale: 1
    m_AlbedoBoost: 1
    m_EnvironmentLightingMode: 0
    m_EnableBakedLightmaps: 1
    m_EnableRealtimeLightmaps: 0
  m_LightmapEditorSettings:
    serializedVersion: 12
    m_Resolution: 2
    m_BakeResolution: 40
    m_AtlasSize: 1024
    m_AO: 0
    m_AOMaxDistance: 1
    m_CompAOExponent: 1
    m_CompAOExponentDirect: 0
    m_ExtractAmbientOcclusion: 0
    m_Padding: 2
    m_LightmapParameters: {{fileID: 0}}
    m_LightmapsBakeMode: 1
    m_TextureCompression: 1
    m_FinalGather: 0
    m_FinalGatherFiltering: 1
    m_FinalGatherRayCount: 256
    m_ReflectionCompression: 2
    m_MixedBakeMode: 2
    m_BakeBackend: 1
    m_PVRSampling: 1
    m_PVRDirectSampleCount: 32
    m_PVRSampleCount: 512
    m_PVRBounces: 2
    m_PVREnvironmentSampleCount: 256
    m_PVREnvironmentReferencePointCount: 2048
    m_PVRFilteringMode: 1
    m_PVRDenoiserTypeDirect: 1
    m_PVRDenoiserTypeIndirect: 1
    m_PVRDenoiserTypeAO: 1
    m_PVRFilterTypeDirect: 0
    m_PVRFilterTypeIndirect: 0
    m_PVRFilterTypeAO: 0
    m_PVREnvironmentMIS: 1
    m_PVRCulling: 1
    m_PVRFilteringGaussRadiusDirect: 1
    m_PVRFilteringGaussRadiusIndirect: 5
    m_PVRFilteringGaussRadiusAO: 2
    m_PVRFilteringAtrousPositionSigmaDirect: 0.5
    m_PVRFilteringAtrousPositionSigmaIndirect: 2
    m_PVRFilteringAtrousPositionSigmaAO: 1
    m_ExportTrainingData: 0
    m_TrainingDataDestination: TrainingData
    m_LightProbeSampleCountMultiplier: 4
  m_LightingDataAsset: {{fileID: 0}}
  m_LightingSettings: {{fileID: 0}}
--- !u!196 &4
NavMeshSettings:
  serializedVersion: 2
  m_ObjectHideFlags: 0
  m_BuildSettings:
    serializedVersion: 2
    agentTypeID: 0
    agentRadius: 0.5
    agentHeight: 2
    agentSlope: 45
    agentClimb: 0.4
    ledgeDropHeight: 0
    maxJumpAcrossDistance: 0
    minRegionArea: 2
    manualCellSize: 0
    cellSize: 0.16666667
    manualTileSize: 0
    tileSize: 256
    accuratePlacement: 0
    maxJobWorkers: 0
    preserveTilesOutsideBounds: 0
    debug:
      m_Flags: 0
  m_NavMeshData: {{fileID: 0}}

# Research Environment Configuration: {self.config.scene_name}
# Description: {self.config.description}

# Physics Settings
Gravity: {self.config.physics_settings['gravity']}
Solver Type: {self.config.physics_settings['solver_type']}
Solver Iteration Count: {self.config.physics_settings['solver_iteration_count']}

# Environment Objects
# (Would normally contain GameObject definitions)
"""
        for obj in self.config.environment_objects:
            scene_content += f"""
# Object: {obj['name']}
Type: {obj['type']}
Position: ({obj['position']['x']}, {obj['position']['y']}, {obj['position']['z']})
"""

        return scene_content
    
    def create_research_task_script(self, task_name: str) -> str:
        """
        Create a C# script for a specific research task.
        
        Args:
            task_name: Name of the research task
            
        Returns:
            Path to the created script
        """
        # Find the task configuration
        task_config = None
        for task in self.config.research_tasks:
            if task['name'] == task_name:
                task_config = task
                break
        
        if not task_config:
            raise ValueError(f"Research task '{task_name}' not found in configuration")
        
        # Create C# script content
        script_content = self._create_research_task_script_content(task_config)
        
        # Write the script
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        script_path = unity_project_path / "Assets" / "Scripts" / f"{task_name}_Task.cs"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created research task script: {script_path}")
        return str(script_path)
    
    def _create_research_task_script_content(self, task_config: Dict) -> str:
        """
        Create content for a research task C# script.
        
        Args:
            task_config: Configuration dictionary for the task
            
        Returns:
            String content of the C# script
        """
        script_content = f"""using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Research Task: {task_config['name']}
/// Description: {task_config['description']}
/// </summary>
public class {task_config['name']}_Task : MonoBehaviour
{{
    [Header("Task Configuration")]
    public string taskName = "{task_config['name']}";
    public string description = "{task_config['description']}";
    
    [Header("Metrics")]
    public float startTime;
    public float completionTime;
    public bool taskCompleted = false;
    public int successCount = 0;
    public int attemptCount = 0;
    
    // List of metrics to track
    private Dictionary<string, float> metrics = new Dictionary<string, float>();
    
    void Start()
    {{
        startTime = Time.time;
        InitializeMetrics();
        Debug.Log($"Research task '{{taskName}}' started.");
    }}
    
    void Update()
    {{
        // Check for task completion conditions
        if (!taskCompleted && CheckTaskCompletion())
        {{
            CompleteTask();
        }}
    }}
    
    private void InitializeMetrics()
    {{
        // Initialize metrics based on configuration
        string[] metricNames = new string[] {{ ";
        for i, metric in enumerate(task_config['metrics']):
            if i > 0:
                script_content += f", ";
            script_content += f"\"{metric}\"";
        script_content += f" }};
        
        foreach (string metric in metricNames)
        {{
            metrics[metric] = 0.0f;
        }}
    }}
    
    private bool CheckTaskCompletion()
    {{
        // Implement task-specific completion logic
        // This is a placeholder implementation
        return false;
    }}
    
    private void CompleteTask()
    {{
        completionTime = Time.time - startTime;
        taskCompleted = true;
        LogTaskResults();
        Debug.Log($"Research task '{{taskName}}' completed in {{completionTime:F2}} seconds.");
    }}
    
    private void LogTaskResults()
    {{
        // Log results to Unity console and potentially to file
        string results = $"Task Results for '{{taskName}}':";
        results += $"\n  Completion Time: {{completionTime:F2}}s";
        results += $"\n  Success Rate: {{(attemptCount > 0 ? (float)successCount / attemptCount * 100 : 0):F1}}%";
        
        foreach (var metric in metrics)
        {{
            results += $"\n  {{metric.Key}}: {{metric.Value}}";
        }}
        
        Debug.Log(results);
        
        // In a real implementation, save results to a persistent storage
        SaveResultsToFile(results);
    }}
    
    private void SaveResultsToFile(string results)
    {{
        // In a real implementation, save to persistent storage
        // This could be a JSON file or other format
    }}
    
    /// <summary>
    /// Called to reset the task to initial state
    /// </summary>
    public void ResetTask()
    {{
        taskCompleted = false;
        successCount = 0;
        attemptCount = 0;
        startTime = Time.time;
        
        // Reset metrics
        foreach (var key in metrics.Keys)
        {{
            metrics[key] = 0.0f;
        }}
        
        Debug.Log($"Research task '{{taskName}}' reset to initial state.");
    }}
}}
"""
        return script_content
    
    def create_sensor_interface_script(self) -> str:
        """
        Create a C# script for sensor interfaces in the research environment.
        
        Returns:
            Path to the created script
        """
        script_content = """using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Sensor Interface Manager for Research Environment
/// Handles data collection from various sensors in the Unity simulation
/// </summary>
public class SensorInterfaceManager : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public Camera rgbCamera;
    public Camera depthCamera;
    public Transform imuTransform;  // Position where IMU sensor is located
    
    [Header("Data Collection")]
    public bool collectData = true;
    public float dataCollectionInterval = 0.1f;
    private float lastCollectionTime;
    
    // Sensor data storage
    private List<SensorDataPoint> sensorDataLog = new List<SensorDataPoint>();
    
    void Start()
    {
        lastCollectionTime = Time.time;
        Debug.Log("Sensor Interface Manager initialized.");
    }
    
    void Update()
    {
        if (collectData && Time.time - lastCollectionTime >= dataCollectionInterval)
        {
            CollectSensorData();
            lastCollectionTime = Time.time;
        }
    }
    
    private void CollectSensorData()
    {
        SensorDataPoint dataPoint = new SensorDataPoint();
        dataPoint.timestamp = Time.time;
        
        // Collect RGB image data
        if (rgbCamera != null)
        {
            dataPoint.rgbImageData = CaptureCameraImage(rgbCamera);
        }
        
        // Collect depth data
        if (depthCamera != null)
        {
            dataPoint.depthImageData = CaptureCameraImage(depthCamera);
        }
        
        // Collect IMU data (simplified)
        if (imuTransform != null)
        {
            dataPoint.imuData = new IMUSensorData
            {
                position = imuTransform.position,
                rotation = imuTransform.rotation,
                velocity = GetComponent<Rigidbody>() ? GetComponent<Rigidbody>().velocity : Vector3.zero,
                angularVelocity = GetComponent<Rigidbody>() ? GetComponent<Rigidbody>().angularVelocity : Vector3.zero
            };
        }
        
        // Add to log
        sensorDataLog.Add(dataPoint);
        
        // In a real implementation, this data would be sent to a data collection system
        Debug.Log($"Collected sensor data at time: {dataPoint.timestamp:F3}");
    }
    
    private Texture2D CaptureCameraImage(Camera cam)
    {
        // Create a temporary render texture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture renderTexture = new RenderTexture(cam.targetTexture.width, cam.targetTexture.height, 24);
        cam.targetTexture = renderTexture;
        cam.Render();
        
        RenderTexture.active = renderTexture;
        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();
        
        // Cleanup
        RenderTexture.active = currentRT;
        Destroy(renderTexture);
        
        return image;
    }
    
    [System.Serializable]
    public class SensorDataPoint
    {
        public float timestamp;
        public Texture2D rgbImageData;
        public Texture2D depthImageData;
        public IMUSensorData imuData;
    }
    
    [System.Serializable]
    public class IMUSensorData
    {
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 velocity;
        public Vector3 angularVelocity;
    }
    
    /// <summary>
    /// Get the collected sensor data
    /// </summary>
    /// <returns>List of sensor data points</returns>
    public List<SensorDataPoint> GetSensorDataLog()
    {
        return sensorDataLog;
    }
    
    /// <summary>
    /// Clear the sensor data log
    /// </summary>
    public void ClearSensorDataLog()
    {
        sensorDataLog.Clear();
        Debug.Log("Sensor data log cleared.");
    }
    
    /// <summary>
    /// Save collected data to persistent storage
    /// </summary>
    public void SaveCollectedData()
    {
        // In a real implementation, save to file or database
        Debug.Log($"Saving {sensorDataLog.Count} sensor data points to persistent storage.");
    }
}
"""
        
        # Write the script
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        script_path = unity_project_path / "Assets" / "Scripts" / "SensorInterfaceManager.cs"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created sensor interface script: {script_path}")
        return str(script_path)
    
    def generate_research_report_template(self) -> str:
        """
        Generate a template for research experiment reports.
        
        Returns:
            Path to the generated report template
        """
        report_template = f"""# Research Experiment Report: {self.config.scene_name}

## 1. Experiment Overview
- **Research Question**: [To be filled by researcher]
- **Hypothesis**: [To be filled by researcher]
- **Environment**: {self.config.scene_name}
- **Description**: {self.config.description}
- **Date**: [To be filled by researcher]
- **Researcher**: [To be filled by researcher]

## 2. Environment Configuration
### Physics Settings
- Gravity: {self.config.physics_settings['gravity']}
- Solver Type: {self.config.physics_settings['solver_type']}
- Solver Iteration Count: {self.config.physics_settings['solver_iteration_count']}

### Environment Objects
[Details of objects in the environment to be filled by researcher]

### Sensors
[Details of sensors used to be filled by researcher]

### Safety Boundaries
[Details of safety boundaries to be filled by researcher]

## 3. Research Tasks Performed
[List of research tasks and their configurations]

## 4. Methodology
[Experimental methodology to be filled by researcher]

## 5. Results
### Quantitative Results
[Quantitative results to be filled by researcher]

### Qualitative Observations
[Qualitative observations to be filled by researcher]

## 6. Analysis
[Analysis of results to be filled by researcher]

## 7. Reproducibility
### System Configuration
[Details of system configuration for reproducibility]

### Steps to Reproduce
[Detailed steps to reproduce the experiment]

## 8. Conclusions
[Conclusions to be filled by researcher]

## 9. Future Work
[Suggestions for future work to be filled by researcher]

## 10. References
[References to be filled by researcher]
"""
        
        # Write the template
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        report_path = unity_project_path / "Assets" / "ResearchData" / "research_report_template.md"
        
        with open(report_path, 'w') as f:
            f.write(report_template)
        
        print(f"Generated research report template: {report_path}")
        return str(report_path)
    
    def export_for_isaac_ros(self) -> str:
        """
        Export the Unity environment configuration in a format compatible with Isaac ROS.
        
        Returns:
            Path to the exported configuration
        """
        # Create Isaac ROS compatible configuration
        isaac_ros_config = {
            "scene_name": self.config.scene_name,
            "description": self.config.description,
            "environment_objects": [],
            "sensors": [],
            "physics_settings": self.config.physics_settings
        }
        
        # Convert environment objects
        for obj in self.config.environment_objects:
            isaac_obj = {
                "name": obj["name"],
                "type": obj["type"],
                "position": obj["position"],
                "physics": {
                    "mass": obj.get("mass", 1.0),
                    "friction": obj.get("material_friction", 0.5)
                }
            }
            if "dimensions" in obj:
                isaac_obj["dimensions"] = obj["dimensions"]
            elif "radius" in obj:
                isaac_obj["radius"] = obj["radius"]
            
            isaac_ros_config["environment_objects"].append(isaac_obj)
        
        # Convert sensors
        for sensor in self.config.sensors:
            isaac_sensor = {
                "sensor_type": sensor["type"],
                "position": sensor["position"],
                "rotation": sensor.get("rotation", {"x": 0, "y": 0, "z": 0}),
                "parameters": {
                    "resolution": sensor.get("resolution"),
                    "fov": sensor.get("fov")
                }
            }
            isaac_ros_config["sensors"].append(isaac_sensor)
        
        # Write Isaac ROS configuration
        unity_project_path = Path(f"unity_research_projects/{self.config.scene_name}")
        config_path = unity_project_path / "Assets" / "ResearchData" / "isaac_ros_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(isaac_ros_config, f, indent=2)
        
        print(f"Exported Isaac ROS compatible configuration: {config_path}")
        return str(config_path)


def create_sample_research_environment():
    """
    Create a sample research environment to demonstrate functionality.
    """
    print("Creating sample Unity research environment...")
    
    # Create a custom configuration
    config = {
        'scene_name': 'sample_manipulation_study',
        'description': 'Study of object manipulation with different physical properties',
        'physics_settings': {
            "gravity": -9.81,
            "solver_type": "SequentialImpulses",
            "solver_iteration_count": 6,
            "sleep_threshold": 0.005,
            "default_contact_offset": 0.01
        },
        'robot_prefabs': ['TurtleBot4'],
        'environment_objects': [
            {
                "name": "table",
                "type": "Static",
                "dimensions": {"x": 1.2, "y": 0.8, "z": 0.75},
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "material_friction": 0.5
            },
            {
                "name": "object1", 
                "type": "Dynamic",
                "shape": "Cube",
                "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                "position": {"x": 0.2, "y": 0.0, "z": 0.15},
                "mass": 0.5,
                "material_friction": 0.3
            },
            {
                "name": "object2",
                "type": "Dynamic", 
                "shape": "Sphere",
                "radius": 0.05,
                "position": {"x": -0.2, "y": 0.0, "z": 0.1},
                "mass": 0.2,
                "material_friction": 0.7
            }
        ],
        'sensors': [
            {
                "type": "RGB_Camera",
                "resolution": {"width": 640, "height": 480},
                "fov": 60,
                "position": {"x": 0.0, "y": -0.5, "z": 1.0},
                "rotation": {"x": -10, "y": 0, "z": 0}
            }
        ],
        'research_tasks': [
            {
                "name": "manipulation_task",
                "description": "Manipulate objects to target positions",
                "object_parameters": [
                    {"shape": "cube", "size": 0.1, "weight": 0.5},
                    {"shape": "sphere", "size": 0.05, "weight": 0.2}
                ],
                "metrics": ["success_rate", "manipulation_time", "energy_consumption"]
            }
        ],
        'safety_boundaries': {
            "workspace_volume": {"min": {"x": -2.0, "y": -2.0, "z": 0.0}, 
                               "max": {"x": 2.0, "y": 2.0, "z": 2.0}},
            "robot_speed_limits": {"linear": 1.0, "angular": 1.5},
            "emergency_stop_conditions": ["collision", "boundary_violation", "timeout"]
        }
    }
    
    # Create YAML config file
    config_path = Path("unity_research_projects") / "sample_config.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create the environment
    env = UnityResearchEnvironment(config_path=str(config_path))
    
    # Create a research task script
    env.create_research_task_script("manipulation_task")
    
    # Create sensor interface script
    env.create_sensor_interface_script()
    
    # Generate research report template
    env.generate_research_report_template()
    
    # Export for Isaac ROS
    env.export_for_isaac_ros()
    
    print(f"Sample research environment created at unity_research_projects/{env.config.scene_name}")
    
    return env


if __name__ == "__main__":
    # Create a sample research environment
    env = create_sample_research_environment()
    print("\nUnity research environment created successfully!")