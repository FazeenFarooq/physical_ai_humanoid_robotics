"""
Research experiment framework for the Physical AI & Humanoid Robotics course.
This module provides tools for researchers to conduct reproducible experiments
and validate new approaches to embodied AI as specified in the data model.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime
import uuid


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    variables: Dict[str, Any]
    constants: Dict[str, Any]
    robot_model: str
    environment: str
    duration: int  # Expected duration in minutes
    required_equipment: List[str]


class ExperimentResult:
    """Results from a completed experiment"""
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.observations = []
        self.raw_data_paths = []
        self.notes = []
        self.success = False
        self.error_message = None
    
    def add_metric(self, name: str, value: Any):
        """Add a metric to the experiment results"""
        self.metrics[name] = value
    
    def add_observation(self, observation: str):
        """Add an observation to the experiment"""
        self.observations.append(observation)
    
    def add_raw_data_path(self, path: str):
        """Add a path to raw data collected during the experiment"""
        self.raw_data_paths.append(path)
    
    def add_note(self, note: str):
        """Add a note about the experiment"""
        self.notes.append(note)


class Experiment(ABC):
    """Abstract base class for research experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result = ExperimentResult(config.experiment_id)
        self.is_running = False
    
    @abstractmethod
    def setup(self):
        """Set up the experiment environment and initial conditions"""
        pass
    
    @abstractmethod
    def run(self):
        """Execute the main experiment procedure"""
        pass
    
    @abstractmethod
    def teardown(self):
        """Clean up after the experiment"""
        pass
    
    def execute(self) -> ExperimentResult:
        """Execute the complete experiment workflow"""
        self.result.start_time = datetime.now()
        self.is_running = True
        
        try:
            self.setup()
            self.run()
            self.teardown()
            self.result.success = True
        except Exception as e:
            self.result.error_message = str(e)
            self.result.success = False
        finally:
            self.result.end_time = datetime.now()
            self.is_running = False
        
        return self.result


class ExperimentFramework:
    """Main framework for managing research experiments"""
    
    def __init__(self, data_dir: str = "experiment_data"):
        self.experiments: Dict[str, Experiment] = {}
        self.data_dir = data_dir
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure the experiment data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def register_experiment(self, experiment: Experiment):
        """Register an experiment with the framework"""
        self.experiments[experiment.config.experiment_id] = experiment
    
    def run_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Run a registered experiment by ID"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not registered")
        
        experiment = self.experiments[experiment_id]
        result = experiment.execute()
        
        # Save experiment results
        self._save_experiment_result(experiment_id, result)
        
        return result
    
    def _save_experiment_result(self, experiment_id: str, result: ExperimentResult):
        """Save experiment results to persistent storage"""
        # Create experiment directory
        exp_dir = os.path.join(self.data_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save result metadata
        result_data = {
            'experiment_id': result.experiment_id,
            'start_time': result.start_time.isoformat() if result.start_time else None,
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'metrics': result.metrics,
            'observations': result.observations,
            'raw_data_paths': result.raw_data_paths,
            'notes': result.notes,
            'success': result.success,
            'error_message': result.error_message
        }
        
        # Write to JSON file
        result_path = os.path.join(exp_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def list_experiments(self) -> List[str]:
        """List all registered experiment IDs"""
        return list(self.experiments.keys())
    
    def get_experiment_result(self, experiment_id: str) -> Optional[Dict]:
        """Retrieve saved results for an experiment"""
        result_path = os.path.join(self.data_dir, experiment_id, 'result.json')
        if not os.path.exists(result_path):
            return None
        
        with open(result_path, 'r') as f:
            return json.load(f)


# Example implementation of a specific experiment
class PerceptionValidationExperiment(Experiment):
    """Example experiment to validate perception algorithms"""
    
    def setup(self):
        """Set up the perception validation experiment"""
        print(f"Setting up perception validation: {self.config.name}")
        # Initialize robot and environment based on config
        # Set parameters for the perception algorithm being tested
    
    def run(self):
        """Run the perception validation experiment"""
        print(f"Running perception validation: {self.config.name}")
        # Execute the experiment protocol
        # Collect data on perception accuracy
        self.result.add_metric("accuracy", 0.92)
        self.result.add_metric("processing_time_ms", 45.2)
        self.result.add_observation("Object detection successful in 95% of frames")
    
    def teardown(self):
        """Clean up after the perception validation experiment"""
        print(f"Cleaning up perception validation: {self.config.name}")
        # Reset robot and environment