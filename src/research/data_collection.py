"""
Experimental Data Collection and Validation Tools

This module provides tools for researchers to collect, manage, and validate 
experimental data in the Physical AI & Humanoid Robotics course.
"""

import os
import json
import yaml
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import cv2
from pathlib import Path

from perception_data import PerceptionData
from action_command import ActionCommand


@dataclass
class ExperimentMetadata:
    """Metadata for a research experiment"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    start_time: datetime
    end_time: Optional[datetime] = None
    researcher: str = "Unknown"
    robot_model: str = "Unknown"
    environment: str = "Unknown"
    hardware_config: str = "Unknown"
    software_stack: str = "Unknown"
    status: str = "in_progress"


class DataCollector:
    """
    Collects experimental data from various sources in a structured format
    """
    
    def __init__(self, experiment_id: str, output_dir: str = "./experiment_data"):
        """
        Initialize the data collector for a specific experiment
        
        Args:
            experiment_id: Unique identifier for the experiment
            output_dir: Directory where collected data will be stored
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        self.metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name="",
            description="",
            hypothesis="",
            start_time=datetime.now(),
            researcher="",
            robot_model="",
            environment=""
        )
        
        # Prepare storage for different data types
        self.perception_data = []
        self.action_commands = []
        self.performance_metrics = {}
        self.system_logs = []
        self.environment_data = {}
        
        # Initialize data counters
        self.perception_count = 0
        self.action_count = 0
        self.metrics_count = 0
        
        # Create subdirectories for different data types
        (self.output_dir / "perception").mkdir(exist_ok=True)
        (self.output_dir / "actions").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "environment").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)

    def set_metadata(self, **kwargs):
        """
        Set metadata fields for the experiment
        
        Args:
            **kwargs: Metadata fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
    
    def collect_perception_data(self, perception_item: PerceptionData) -> str:
        """
        Collect perception data and save it to the appropriate directory
        
        Args:
            perception_item: PerceptionData object to store
            
        Returns:
            File path of the saved data
        """
        # Create filename with timestamp and counter
        filename = f"perception_{self.perception_count:06d}_{perception_item.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / "perception" / filename
        
        # Prepare data dictionary
        data_dict = {
            "id": perception_item.id,
            "timestamp": perception_item.timestamp.isoformat(),
            "sensor_type": perception_item.sensor_type,
            "data_shape": perception_item.data.shape if hasattr(perception_item.data, 'shape') else str(type(perception_item.data)),
            "environment": perception_item.environment,
            "robot_state": perception_item.robot_state,
            "annotations": perception_item.annotations,
            "source": perception_item.source,
            "quality_score": perception_item.quality_score
        }
        
        # Save the data
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
        
        self.perception_data.append(filepath)
        self.perception_count += 1
        
        return str(filepath)
    
    def collect_action_data(self, action_command: ActionCommand) -> str:
        """
        Collect action command data and save it to the appropriate directory
        
        Args:
            action_command: ActionCommand object to store
            
        Returns:
            File path of the saved data
        """
        # Create filename with timestamp and counter
        filename = f"action_{self.action_count:06d}_{action_command.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / "actions" / filename
        
        # Prepare data dictionary
        data_dict = {
            "id": action_command.id,
            "type": action_command.type,
            "parameters": action_command.parameters,
            "priority": action_command.priority,
            "timestamp": action_command.timestamp.isoformat(),
            "status": action_command.status,
            "executor": action_command.executor,
            "dependencies": action_command.dependencies
        }
        
        # Save the data
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
        
        self.action_commands.append(filepath)
        self.action_count += 1
        
        return str(filepath)
    
    def collect_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Collect performance metrics and save them
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            File path of the saved metrics
        """
        # Create filename with timestamp and counter
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"metrics_{self.metrics_count:06d}_{timestamp}.json"
        filepath = self.output_dir / "metrics" / filename
        
        # Add timestamp to metrics
        metrics_with_timestamp = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        # Save the metrics
        with open(filepath, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=2, default=str)
        
        self.performance_metrics[filepath] = metrics
        self.metrics_count += 1
        
        return str(filepath)
    
    def collect_system_log(self, log_entry: str, level: str = "INFO") -> str:
        """
        Collect system log entry and save it
        
        Args:
            log_entry: The log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            
        Returns:
            File path of the saved log
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        filename = f"log_{timestamp}_{level.lower()}.txt"
        filepath = self.output_dir / "logs" / filename
        
        # Format the log entry
        formatted_log = f"[{timestamp}] {level}: {log_entry}\n"
        
        # Write the log entry
        with open(filepath, 'w') as f:
            f.write(formatted_log)
        
        self.system_logs.append(filepath)
        
        return str(filepath)
    
    def collect_environment_state(self, env_data: Dict[str, Any]) -> str:
        """
        Collect environment state data
        
        Args:
            env_data: Dictionary containing environment state information
            
        Returns:
            File path of the saved environment data
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"environment_{timestamp}.json"
        filepath = self.output_dir / "environment" / filename
        
        # Add timestamp to the environment data
        env_data_with_timestamp = {
            "timestamp": timestamp,
            "environment_state": env_data
        }
        
        # Save the environment data
        with open(filepath, 'w') as f:
            json.dump(env_data_with_timestamp, f, indent=2, default=str)
        
        self.environment_data[filepath] = env_data
        
        return str(filepath)
    
    def capture_image(self, image_data, prefix: str = "capture") -> str:
        """
        Capture and save an image
        
        Args:
            image_data: Image data (numpy array or similar)
            prefix: Prefix for the filename
            
        Returns:
            File path of the saved image
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        filename = f"{prefix}_{timestamp}.png"
        filepath = self.output_dir / "images" / filename
        
        # Save image using OpenCV (assumes BGR format)
        cv2.imwrite(str(filepath), image_data)
        
        return str(filepath)
    
    def finalize_experiment(self) -> str:
        """
        Finalize the experiment and save metadata
        
        Returns:
            Path to the metadata file
        """
        # Update end time
        self.metadata.end_time = datetime.now()
        self.metadata.status = "completed"
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        metadata_dict = {
            "experiment_id": self.metadata.experiment_id,
            "name": self.metadata.name,
            "description": self.metadata.description,
            "hypothesis": self.metadata.hypothesis,
            "start_time": self.metadata.start_time.isoformat(),
            "end_time": self.metadata.end_time.isoformat() if self.metadata.end_time else None,
            "researcher": self.metadata.researcher,
            "robot_model": self.metadata.robot_model,
            "environment": self.metadata.environment,
            "hardware_config": self.metadata.hardware_config,
            "software_stack": self.metadata.software_stack,
            "status": self.metadata.status
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Create summary statistics
        summary = {
            "experiment_id": self.experiment_id,
            "total_perception_samples": len(self.perception_data),
            "total_action_commands": len(self.action_commands),
            "total_performance_metrics": len(self.performance_metrics),
            "total_logs": len(self.system_logs),
            "data_collection_duration": str(self.metadata.end_time - self.metadata.start_time),
            "output_directory": str(self.output_dir)
        }
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(metadata_path)


class DataValidator:
    """
    Validates collected experimental data for quality and consistency
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the data validator
        
        Args:
            experiment_dir: Directory containing experiment data
        """
        self.experiment_dir = Path(experiment_dir)
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load the experiment metadata
        
        Returns:
            Metadata dictionary
        """
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def validate_perception_data(self) -> Dict[str, Any]:
        """
        Validate perception data quality
        
        Returns:
            Validation results
        """
        perception_dir = self.experiment_dir / "perception"
        if not perception_dir.exists():
            return {"error": "Perception data directory does not exist"}
        
        validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "quality_scores": [],
            "sensor_types": {},
            "time_intervals": []
        }
        
        files = list(perception_dir.glob("*.json"))
        validation_results["total_files"] = len(files)
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract quality score if available
                if 'quality_score' in data:
                    validation_results["quality_scores"].append(data["quality_score"])
                
                # Count sensor types
                if 'sensor_type' in data:
                    sensor_type = data["sensor_type"]
                    validation_results["sensor_types"][sensor_type] = validation_results["sensor_types"].get(sensor_type, 0) + 1
                
                validation_results["valid_files"] += 1
            
            except Exception as e:
                print(f"Error validating {file_path}: {e}")
                validation_results["invalid_files"] += 1
        
        # Calculate average quality score if we have data
        if validation_results["quality_scores"]:
            avg_quality = sum(validation_results["quality_scores"]) / len(validation_results["quality_scores"])
            validation_results["average_quality_score"] = avg_quality
        
        return validation_results
    
    def validate_action_data(self) -> Dict[str, Any]:
        """
        Validate action command data
        
        Returns:
            Validation results
        """
        action_dir = self.experiment_dir / "actions"
        if not action_dir.exists():
            return {"error": "Action data directory does not exist"}
        
        validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "action_types": {},
            "action_status": {},
            "execution_times": []
        }
        
        files = list(action_dir.glob("*.json"))
        validation_results["total_files"] = len(files)
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Count action types
                if 'type' in data:
                    action_type = data["type"]
                    validation_results["action_types"][action_type] = validation_results["action_types"].get(action_type, 0) + 1
                
                # Count action status
                if 'status' in data:
                    status = data["status"]
                    validation_results["action_status"][status] = validation_results["action_status"].get(status, 0) + 1
                
                validation_results["valid_files"] += 1
            
            except Exception as e:
                print(f"Error validating {file_path}: {e}")
                validation_results["invalid_files"] += 1
        
        return validation_results
    
    def validate_performance_metrics(self) -> Dict[str, Any]:
        """
        Validate performance metrics data
        
        Returns:
            Validation results
        """
        metrics_dir = self.experiment_dir / "metrics"
        if not metrics_dir.exists():
            return {"error": "Performance metrics directory does not exist"}
        
        validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "metrics_summary": {}
        }
        
        files = list(metrics_dir.glob("*.json"))
        validation_results["total_files"] = len(files)
        
        all_metrics = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'metrics' in data:
                    all_metrics.append(data['metrics'])
                    validation_results["valid_files"] += 1
                else:
                    validation_results["invalid_files"] += 1
            
            except Exception as e:
                print(f"Error validating {file_path}: {e}")
                validation_results["invalid_files"] += 1
        
        # Calculate summary statistics for common metrics
        if all_metrics:
            # Find common keys across all metric dictionaries
            all_keys = set()
            for metrics in all_metrics:
                all_keys.update(metrics.keys())
            
            # Calculate statistics for each metric
            for key in all_keys:
                values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    validation_results["metrics_summary"][key] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "std": float(np.std(values)) if len(values) > 1 else 0
                    }
        
        return validation_results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report
        
        Returns:
            Validation report
        """
        report = {
            "experiment_id": self.metadata.get("experiment_id", "Unknown"),
            "validation_timestamp": datetime.now().isoformat(),
            "perception_validation": self.validate_perception_data(),
            "action_validation": self.validate_action_data(),
            "performance_validation": self.validate_performance_metrics()
        }
        
        # Overall quality assessment
        perc_valid = report["perception_validation"].get("valid_files", 0)
        perc_total = report["perception_validation"].get("total_files", 1)
        action_valid = report["action_validation"].get("valid_files", 0)
        action_total = report["action_validation"].get("total_files", 1)
        
        perc_quality = perc_valid / perc_total if perc_total > 0 else 0
        action_quality = action_valid / action_total if action_total > 0 else 0
        
        report["overall_quality_assessment"] = {
            "perception_data_quality": perc_quality,
            "action_data_quality": action_quality,
            "overall_data_quality": (perc_quality + action_quality) / 2
        }
        
        # Quality rating
        avg_quality = report["overall_quality_assessment"]["overall_data_quality"]
        if avg_quality >= 0.95:
            report["quality_rating"] = "Excellent"
        elif avg_quality >= 0.85:
            report["quality_rating"] = "Good"
        elif avg_quality >= 0.70:
            report["quality_rating"] = "Fair"
        else:
            report["quality_rating"] = "Poor"
        
        return report


def create_sample_experiment():
    """
    Creates a sample experiment to demonstrate the data collection tools
    """
    # Create a data collector for a sample experiment
    collector = DataCollector("sample_experiment_001")
    
    # Set metadata for the experiment
    collector.set_metadata(
        name="Sample Perception-Action Correlation Experiment",
        description="Testing correlation between visual perception and navigation actions",
        hypothesis="Higher quality visual perception leads to more efficient navigation actions",
        researcher="AI Researcher",
        robot_model="TurtleBot 4",
        environment="Gazebo Home Environment"
    )
    
    # Simulate collecting some data
    import random
    
    print("Collecting sample perception and action data...")
    
    for i in range(10):
        # Simulate perception data
        from datetime import timedelta
        timestamp = datetime.now() - timedelta(seconds=random.randint(0, 300))
        
        # Create a mock PerceptionData object
        class MockPerceptionData:
            def __init__(self, ts, sensor_type, env, robot_state, quality):
                self.id = f"perception_{i:03d}"
                self.timestamp = ts
                self.sensor_type = sensor_type
                self.data = np.random.rand(480, 640, 3)  # Simulated image data
                self.environment = env
                self.robot_state = robot_state
                self.annotations = ["obstacle", "wall"] if random.random() > 0.5 else ["clear_path"]
                self.source = "simulation" if random.random() > 0.3 else "physical"
                self.quality_score = quality
        
        perception_data = MockPerceptionData(
            timestamp,
            "RGB-D" if random.random() > 0.5 else "LiDAR",
            "home_environment",
            f"position_{i}",
            random.uniform(0.5, 1.0)  # Quality score between 0.5 and 1.0
        )
        
        collector.collect_perception_data(perception_data)
        
        # Simulate action data
        class MockActionCommand:
            def __init__(self, cmd_type, timestamp):
                self.id = f"action_{i:03d}"
                self.type = cmd_type
                self.parameters = {"speed": random.uniform(0.1, 0.5), "direction": random.uniform(-1.0, 1.0)}
                self.priority = 1
                self.timestamp = timestamp
                self.status = "completed"
                self.executor = "navigation_controller"
                self.dependencies = []
        
        action_command = MockActionCommand(
            "navigation" if random.random() > 0.3 else "manipulation",
            timestamp
        )
        
        collector.collect_action_data(action_command)
        
        # Collect some performance metrics
        metrics = {
            "perception_latency": random.uniform(20, 100),  # ms
            "action_execution_time": random.uniform(500, 1500),  # ms
            "battery_level": random.uniform(0.7, 0.95),
            "cpu_usage": random.uniform(30, 70),
            "memory_usage": random.uniform(40, 80)
        }
        collector.collect_performance_metrics(metrics)
    
    # Collect environment state
    env_state = {
        "map_coverage": random.uniform(0.6, 0.95),
        "obstacle_density": random.uniform(0.1, 0.4),
        "lighting_conditions": "indoor_normal" if random.random() > 0.5 else "indoor_dim"
    }
    collector.collect_environment_state(env_state)
    
    # Finalize the experiment
    metadata_path = collector.finalize_experiment()
    print(f"Sample experiment completed. Metadata saved to: {metadata_path}")
    
    # Validate the collected data
    validator = DataValidator(collector.output_dir)
    report = validator.generate_validation_report()
    
    print("\nValidation Report:")
    print(json.dumps(report, indent=2, default=str))
    
    return collector.output_dir


if __name__ == "__main__":
    # Create a sample experiment to demonstrate functionality
    output_dir = create_sample_experiment()
    print(f"\nSample experiment data saved to: {output_dir}")