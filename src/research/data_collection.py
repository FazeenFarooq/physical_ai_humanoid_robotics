"""
Experimental data collection and validation tools for research in Physical AI.

This module provides tools for collecting, storing, validating, and analyzing
experimental data in the context of Physical AI and humanoid robotics research.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
from collections import defaultdict


class DataType(Enum):
    """Enum for different types of experimental data."""
    SENSOR_DATA = "sensor_data"
    PERCEPTION_OUTPUT = "perception_output"
    CONTROL_COMMANDS = "control_commands"
    ROBOT_STATE = "robot_state"
    ENVIRONMENT_STATE = "environment_state"
    USER_INTERACTION = "user_interaction"
    SYSTEM_METRICS = "system_metrics"


@dataclass
class DataPoint:
    """Represents a single data point in an experiment."""
    timestamp: datetime
    data_type: DataType
    data: Any
    experiment_id: str
    participant_id: Optional[str] = None
    robot_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    validation_score: Optional[float] = None


class DataCollector:
    """Collects experimental data for research validation."""
    
    def __init__(self, output_dir: str = "data/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, List[DataPoint]] = defaultdict(list)
        self.current_experiment_id = None
    
    def start_experiment(self, experiment_id: str, description: str = ""):
        """Start a new experiment session."""
        self.current_experiment_id = experiment_id
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Create metadata file for the experiment
        metadata = {
            "experiment_id": experiment_id,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "data_points_count": 0
        }
        
        with open(experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def record_data(self, data_type: DataType, data: Any, 
                   participant_id: Optional[str] = None,
                   robot_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a data point in the current experiment."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment first.")
        
        data_point = DataPoint(
            timestamp=datetime.now(),
            data_type=data_type,
            data=data,
            experiment_id=self.current_experiment_id,
            participant_id=participant_id,
            robot_id=robot_id,
            metadata=metadata or {}
        )
        
        self.experiments[self.current_experiment_id].append(data_point)
        self._save_data_point(data_point)
        
        # Update experiment metadata
        self._update_experiment_metadata(self.current_experiment_id)
        
        # Generate and return a unique ID for this data point
        data_id = hashlib.md5(
            f"{data_point.timestamp.isoformat()}{str(data)}".encode()
        ).hexdigest()[:12]
        return data_id
    
    def _save_data_point(self, data_point: DataPoint):
        """Save a data point to the appropriate file based on its type."""
        experiment_dir = self.output_dir / data_point.experiment_id
        
        # Create a filename based on the data type and timestamp
        timestamp_str = data_point.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp_str}_{data_point.data_type.value}_{len(self.experiments[data_point.experiment_id]):06d}.json"
        filepath = experiment_dir / filename
        
        # Prepare data for serialization
        data_dict = {
            "timestamp": data_point.timestamp.isoformat(),
            "data_type": data_point.data_type.value,
            "data": self._serialize_data(data_point.data),
            "experiment_id": data_point.experiment_id,
            "participant_id": data_point.participant_id,
            "robot_id": data_point.robot_id,
            "metadata": data_point.metadata,
            "validation_score": data_point.validation_score
        }
        
        with open(filepath, "w") as f:
            json.dump(data_dict, f, indent=2, default=str)
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict()
        elif isinstance(data, Path):
            return str(data)
        else:
            # For other data types, try to serialize them, or store as string representation
            try:
                json.dumps(data)  # Test if it's JSON serializable
                return data
            except TypeError:
                # If not serializable, store a string representation
                return str(data)
    
    def _update_experiment_metadata(self, experiment_id: str):
        """Update the metadata file for an experiment."""
        experiment_dir = self.output_dir / experiment_id
        metadata_path = experiment_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "experiment_id": experiment_id,
                "start_time": datetime.now().isoformat(),
                "data_points_count": 0
            }
        
        metadata["data_points_count"] = len(self.experiments[experiment_id])
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


class DataValidator:
    """Validates experimental data for consistency and accuracy."""
    
    def __init__(self):
        self.validation_rules: Dict[DataType, List[callable]] = defaultdict(list)
        self.load_default_validation_rules()
    
    def load_default_validation_rules(self):
        """Load default validation rules for different data types."""
        # Validation rules for sensor data
        self.validation_rules[DataType.SENSOR_DATA] = [
            self._check_sensor_data_completeness,
            self._check_sensor_data_ranges
        ]
        
        # Validation rules for perception output
        self.validation_rules[DataType.PERCEPTION_OUTPUT] = [
            self._check_detection_confidence,
            self._check_object_class_validity
        ]
        
        # Validation rules for control commands
        self.validation_rules[DataType.CONTROL_COMMANDS] = [
            self._check_command_validity,
            self._check_command_bounds
        ]
        
        # Validation rules for robot state
        self.validation_rules[DataType.ROBOT_STATE] = [
            self._check_joint_position_ranges,
            self._check_robot_stability
        ]
        
        # Validation rules for environment state
        self.validation_rules[DataType.ENVIRONMENT_STATE] = [
            self._check_obstacle_positions,
            self._check_environment_consistency
        ]
    
    def validate_data_point(self, data_point: DataPoint) -> float:
        """Validate a single data point and return a validation score (0.0 to 1.0)."""
        if data_point.data_type not in self.validation_rules:
            # Unknown data type, give it a neutral score
            return 0.5
        
        rules = self.validation_rules[data_point.data_type]
        passed_rules = 0
        
        for rule in rules:
            try:
                if rule(data_point):
                    passed_rules += 1
            except Exception as e:
                print(f"Validation rule {rule.__name__} failed: {e}")
        
        score = passed_rules / len(rules) if rules else 1.0
        data_point.validation_score = score
        return score
    
    def _check_sensor_data_completeness(self, data_point: DataPoint) -> bool:
        """Check if sensor data has required fields."""
        if not isinstance(data_point.data, dict):
            return False
        
        # Example: Check if camera image has required fields
        if 'image' in data_point.data:
            return all(key in data_point.data for key in ['timestamp', 'image', 'camera_id'])
        
        # Example: Check if LiDAR data has required fields
        if 'pointcloud' in data_point.data:
            return all(key in data_point.data for key in ['timestamp', 'pointcloud', 'sensor_id'])
        
        return True
    
    def _check_sensor_data_ranges(self, data_point: DataPoint) -> bool:
        """Check if sensor data values are within expected ranges."""
        if not isinstance(data_point.data, dict):
            return False
        
        # Example: Check if depth values are reasonable
        if 'depth' in data_point.data:
            depth_data = data_point.data['depth']
            if isinstance(depth_data, (list, np.ndarray)):
                return all(0.1 <= val <= 10.0 for val in depth_data if isinstance(val, (int, float)))
        
        # Example: Check if IMU values are reasonable
        if 'imu' in data_point.data:
            imu_data = data_point.data['imu']
            if isinstance(imu_data, dict):
                return (abs(imu_data.get('accel_x', 0)) <= 20.0 and
                        abs(imu_data.get('accel_y', 0)) <= 20.0 and
                        abs(imu_data.get('accel_z', 9.81)) <= 30.0)
        
        return True
    
    def _check_detection_confidence(self, data_point: DataPoint) -> bool:
        """Check if object detection confidences are within valid range."""
        if not isinstance(data_point.data, dict) or 'objects' not in data_point.data:
            return True  # Not a detection output, so pass validation
        
        objects = data_point.data['objects']
        if not isinstance(objects, list):
            return False
        
        return all(0.0 <= obj.get('confidence', 1.0) <= 1.0 for obj in objects)
    
    def _check_object_class_validity(self, data_point: DataPoint) -> bool:
        """Check if detected object classes are valid."""
        if not isinstance(data_point.data, dict) or 'objects' not in data_point.data:
            return True  # Not a detection output, so pass validation
        
        objects = data_point.data['objects']
        if not isinstance(objects, list):
            return False
        
        # Valid object classes for our robotics domain
        valid_classes = {
            'chair', 'table', 'cup', 'bottle', 'person', 'door', 'wall', 'floor',
            'robot', 'control_panel', 'button', 'obstacle'
        }
        
        return all(obj.get('class_name', '').lower() in valid_classes for obj in objects)
    
    def _check_command_validity(self, data_point: DataPoint) -> bool:
        """Check if control commands are valid."""
        if not isinstance(data_point.data, dict):
            return False
        
        # Check if it has required command fields
        return 'command_type' in data_point.data and 'parameters' in data_point.data
    
    def _check_command_bounds(self, data_point: DataPoint) -> bool:
        """Check if control command parameters are within safe bounds."""
        if not isinstance(data_point.data, dict) or 'parameters' not in data_point.data:
            return True  # No parameters to check
        
        params = data_point.data['parameters']
        
        # Check velocity bounds for navigation commands
        if data_point.data.get('command_type') == 'navigation':
            vel = params.get('linear_velocity', 0)
            return -1.0 <= vel <= 1.0  # Max 1 m/s for safety
        
        # Check joint position bounds for manipulation commands
        if data_point.data.get('command_type') == 'manipulation':
            joints = params.get('joint_positions', [])
            return all(-3.14 <= pos <= 3.14 for pos in joints)  # Within reasonable joint limits
        
        return True
    
    def _check_joint_position_ranges(self, data_point: DataPoint) -> bool:
        """Check if robot joint positions are within valid ranges."""
        if not isinstance(data_point.data, dict):
            return False
        
        if 'joint_positions' not in data_point.data:
            return True  # No joint positions to check
        
        joint_positions = data_point.data['joint_positions']
        if isinstance(joint_positions, (list, np.ndarray)):
            # Check if all joint positions are within reasonable limits (example: -π to π)
            return all(-3.14159 <= pos <= 3.14159 for pos in joint_positions)
        elif isinstance(joint_positions, dict):
            # If it's a dictionary of joint_name -> position
            return all(-3.14159 <= pos <= 3.14159 for pos in joint_positions.values())
        
        return False
    
    def _check_robot_stability(self, data_point: DataPoint) -> bool:
        """Check if robot state indicates stable operation."""
        if not isinstance(data_point.data, dict):
            return False
        
        # Check center of mass position relative to support polygon for stability
        if 'com_position' in data_point.data and 'support_polygon' in data_point.data:
            # Simplified stability check: CoM should be within support polygon
            com = data_point.data['com_position']
            support = data_point.data['support_polygon']
            
            # This is a simplified check - a real implementation would use geometric algorithms
            return True  # Placeholder for actual stability check
        
        return True
    
    def _check_obstacle_positions(self, data_point: DataPoint) -> bool:
        """Check if obstacle positions are reasonable."""
        if not isinstance(data_point.data, dict) or 'obstacles' not in data_point.data:
            return True  # No obstacles to check
        
        obstacles = data_point.data['obstacles']
        if not isinstance(obstacles, list):
            return False
        
        # Check that all obstacle positions have coordinates
        return all(
            'position' in obs and 'x' in obs['position'] and 'y' in obs['position']
            for obs in obstacles
        )
    
    def _check_environment_consistency(self, data_point: DataPoint) -> bool:
        """Check if environment state is consistent."""
        if not isinstance(data_point.data, dict):
            return False
        
        # Example consistency check: map should have corresponding objects
        if 'map' in data_point.data and 'objects' in data_point.data:
            # A real implementation would check that objects are within map boundaries
            return True
        
        return True


class DataAnalyzer:
    """Analyzes collected experimental data for research validation."""
    
    def __init__(self, data_dir: str = "data/experiments"):
        self.data_dir = Path(data_dir)
        self.collector = DataCollector(output_dir=data_dir)
        self.validator = DataValidator()
    
    def load_experiment_data(self, experiment_id: str) -> List[DataPoint]:
        """Load all data points for a specific experiment."""
        experiment_path = self.data_dir / experiment_id
        
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment data directory {experiment_path} not found")
        
        data_points = []
        for data_file in experiment_path.glob("*.json"):
            if data_file.name == "metadata.json":
                continue  # Skip metadata file
            
            with open(data_file, 'r') as f:
                data_dict = json.load(f)
            
            # Convert dictionary back to DataPoint
            data_point = DataPoint(
                timestamp=datetime.fromisoformat(data_dict['timestamp']),
                data_type=DataType(data_dict['data_type']),
                data=data_dict['data'],
                experiment_id=data_dict['experiment_id'],
                participant_id=data_dict.get('participant_id'),
                robot_id=data_dict.get('robot_id'),
                metadata=data_dict.get('metadata'),
                validation_score=data_dict.get('validation_score')
            )
            data_points.append(data_point)
        
        return data_points
    
    def compute_experiment_metrics(self, experiment_id: str) -> Dict[str, float]:
        """Compute key metrics for an experiment."""
        data_points = self.load_experiment_data(experiment_id)
        
        if not data_points:
            return {}
        
        # Compute validation score statistics
        validation_scores = [dp.validation_score for dp in data_points if dp.validation_score is not None]
        avg_validation_score = np.mean(validation_scores) if validation_scores else 0.0
        
        # Compute data type distribution
        type_counts = defaultdict(int)
        for dp in data_points:
            type_counts[dp.data_type.value] += 1
        
        # Compute time-based metrics
        timestamps = [dp.timestamp for dp in data_points]
        duration = (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0.0
        
        metrics = {
            "total_data_points": len(data_points),
            "avg_validation_score": avg_validation_score,
            "experiment_duration_seconds": duration,
            "data_type_distribution": dict(type_counts)
        }
        
        return metrics
    
    def generate_validation_report(self, experiment_id: str) -> str:
        """Generate a validation report for an experiment."""
        data_points = self.load_experiment_data(experiment_id)
        report_parts = [
            f"Validation Report for Experiment: {experiment_id}",
            "=" * 50,
            f"Total Data Points: {len(data_points)}",
            ""
        ]
        
        # Validate each data point and collect statistics
        scores = []
        for dp in data_points:
            score = self.validator.validate_data_point(dp)
            scores.append(score)
        
        if scores:
            avg_score = np.mean(scores)
            report_parts.append(f"Average Validation Score: {avg_score:.3f}")
            report_parts.append(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
            report_parts.append("")
        
        # Add data type breakdown
        type_counts = defaultdict(int)
        type_scores = defaultdict(list)
        
        for dp in data_points:
            type_counts[dp.data_type.value] += 1
            if dp.validation_score is not None:
                type_scores[dp.data_type.value].append(dp.validation_score)
        
        report_parts.append("Data Type Breakdown:")
        for data_type, count in type_counts.items():
            type_avg_score = np.mean(type_scores[data_type]) if type_scores[data_type] else 0.0
            report_parts.append(f"  {data_type}: {count} points (avg score: {type_avg_score:.3f})")
        
        return "\n".join(report_parts)


def main():
    """Example usage of the data collection and validation tools."""
    # Initialize the tools
    collector = DataCollector()
    validator = DataValidator()
    analyzer = DataAnalyzer()
    
    # Start an experiment
    collector.start_experiment("exp_001", "Testing robot navigation in dynamic environment")
    
    # Simulate collecting some data
    for i in range(10):
        # Collect sensor data
        sensor_data = {
            'timestamp': datetime.now().isoformat(),
            'camera_id': 'front_camera',
            'image_shape': [480, 640, 3],
            'depth': [1.2, 1.5, 2.1, 3.0]  # Example depth readings
        }
        collector.record_data(
            DataType.SENSOR_DATA,
            sensor_data,
            participant_id="P001",
            robot_id="R001",
            metadata={"trial": i}
        )
        
        # Collect robot state
        robot_state = {
            'timestamp': datetime.now().isoformat(),
            'position': {'x': i * 0.1, 'y': 0.0, 'z': 0.0},
            'joint_positions': [0.1 * i] * 5,  # Example joint positions
            'battery_level': 100.0 - (i * 0.5)
        }
        collector.record_data(
            DataType.ROBOT_STATE,
            robot_state,
            participant_id="P001",
            robot_id="R001",
            metadata={"trial": i}
        )
    
    # Generate a validation report for the experiment
    report = analyzer.generate_validation_report("exp_001")
    print(report)


if __name__ == "__main__":
    main()