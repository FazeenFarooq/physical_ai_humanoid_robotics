"""
Reproducibility tools for research validation in Physical AI.

This module provides tools for creating reproducible experiments in the 
Physical AI & Humanoid Robotics course, including experiment tracking,
environment capture, and replication utilities.
"""

import os
import sys
import json
import yaml
import hashlib
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import pickle
import importlib.util
from contextlib import contextmanager


@dataclass
class ReproducibilityMetadata:
    """Metadata required for experiment reproducibility."""
    experiment_id: str
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    description: Optional[str] = None
    author: Optional[str] = None
    hardware_config: Optional[Dict[str, Any]] = None
    software_environment: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, str]] = None
    random_seeds: Optional[Dict[str, int]] = None
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    artifacts: Optional[List[str]] = None
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    platform_info: Optional[Dict[str, str]] = None


class ReproducibilityManager:
    """Manages reproducibility for research experiments."""
    
    def __init__(self, output_dir: str = "research/reproducibility"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Optional[ReproducibilityMetadata] = None
        self.experiment_running = False
    
    def start_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ReproducibilityMetadata:
        """Start a new reproducible experiment."""
        self.experiment_running = True
        
        # Capture the current environment and system state
        software_env = self._capture_software_environment()
        hardware_config = self._capture_hardware_config()
        dependencies = self._capture_dependencies()
        platform_info = self._capture_platform_info()
        git_info = self._capture_git_info()
        
        # Set random seeds for reproducibility (using experiment ID as base)
        random_seeds = self._set_random_seeds(experiment_id)
        
        # Create metadata object
        self.metadata = ReproducibilityMetadata(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
            description=description,
            author=author,
            hardware_config=hardware_config,
            software_environment=software_env,
            dependencies=dependencies,
            random_seeds=random_seeds,
            parameters=parameters or {},
            platform_info=platform_info,
            git_commit_hash=git_info.get('commit_hash'),
            git_branch=git_info.get('branch')
        )
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        print(f"Started reproducible experiment: {experiment_id}")
        return self.metadata
    
    def end_experiment(self, results: Optional[Dict[str, Any]] = None, 
                      artifacts: Optional[List[str]] = None) -> str:
        """End the current experiment and save metadata."""
        if not self.experiment_running or not self.metadata:
            raise RuntimeError("No active experiment to end")
        
        self.metadata.end_time = datetime.now()
        self.metadata.results = results or {}
        self.metadata.artifacts = artifacts or []
        
        # Save metadata to file
        exp_dir = self.output_dir / self.metadata.experiment_id
        metadata_path = exp_dir / "reproducibility_metadata.json"
        
        # Prepare metadata for serialization
        metadata_dict = self._metadata_to_dict(self.metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        self.experiment_running = False
        
        print(f"Ended experiment: {self.metadata.experiment_id}")
        print(f"Metadata saved to: {metadata_path}")
        
        return str(metadata_path)
    
    def _metadata_to_dict(self, metadata: ReproducibilityMetadata) -> Dict[str, Any]:
        """Convert ReproducibilityMetadata to a serializable dictionary."""
        return {
            'experiment_id': metadata.experiment_id,
            'experiment_name': metadata.experiment_name,
            'start_time': metadata.start_time.isoformat(),
            'end_time': metadata.end_time.isoformat() if metadata.end_time else None,
            'description': metadata.description,
            'author': metadata.author,
            'hardware_config': metadata.hardware_config,
            'software_environment': metadata.software_environment,
            'dependencies': metadata.dependencies,
            'random_seeds': metadata.random_seeds,
            'parameters': metadata.parameters,
            'results': metadata.results,
            'artifacts': metadata.artifacts,
            'git_commit_hash': metadata.git_commit_hash,
            'git_branch': metadata.git_branch,
            'platform_info': metadata.platform_info
        }
    
    def _capture_software_environment(self) -> Dict[str, Any]:
        """Capture the current software environment."""
        return {
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler(),
            'os_name': os.name,
            'platform_system': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
        }
    
    def _capture_hardware_config(self) -> Dict[str, Any]:
        """Capture the current hardware configuration."""
        try:
            # CPU info
            cpu_info = {
                'processor': platform.processor(),
                'machine': platform.machine(),
                'architecture': platform.architecture(),
            }
            
            # Memory info (approximate)
            try:
                import psutil
                memory_info = {
                    'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                }
                cpu_info.update(memory_info)
            except ImportError:
                print("psutil not available, skipping detailed memory info")
            
            # GPU info (if available)
            try:
                gpu_info = self._get_gpu_info()
                cpu_info.update(gpu_info)
            except Exception as e:
                print(f"Could not capture GPU info: {e}")
            
            return cpu_info
        except Exception as e:
            print(f"Error capturing hardware config: {e}")
            return {"error": str(e)}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        gpu_info = {}
        
        # Try to get NVIDIA GPU info using nvidia-ml-py3
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info['nvidia_gpu_count'] = device_count
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info[f'gpu_{i}_name'] = name
                gpu_info[f'gpu_{i}_memory_total_mb'] = round(memory_info.total / (1024**2), 2)
                gpu_info[f'gpu_{i}_memory_used_mb'] = round(memory_info.used / (1024**2), 2)
                gpu_info[f'gpu_{i}_utilization_gpu_percent'] = utilization.gpu
                gpu_info[f'gpu_{i}_utilization_memory_percent'] = utilization.memory
            
            pynvml.nvmlShutdown()
        except ImportError:
            print("pynvml not available, skipping NVIDIA GPU info")
        except Exception as e:
            print(f"Error getting NVIDIA GPU info: {e}")
        
        # Try to get general GPU info using pyopencl
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            gpu_info['opencl_platform_count'] = len(platforms)
            
            for i, platform in enumerate(platforms):
                gpu_info[f'opencl_platform_{i}_name'] = platform.name
                devices = platform.get_devices()
                gpu_info[f'opencl_platform_{i}_device_count'] = len(devices)
        except ImportError:
            print("pyopencl not available, skipping OpenCL GPU info")
        except Exception as e:
            print(f"Error getting OpenCL GPU info: {e}")
        
        return gpu_info
    
    def _capture_dependencies(self) -> Dict[str, str]:
        """Capture the current Python dependencies."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format', 'freeze'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                deps = {}
                for line in result.stdout.strip().split('\n'):
                    if line and '==' in line:
                        name, version = line.split('==', 1)
                        deps[name] = version
                return deps
            else:
                print(f"Error running pip list: {result.stderr}")
                return {}
        except Exception as e:
            print(f"Error capturing dependencies: {e}")
            return {}
    
    def _capture_platform_info(self) -> Dict[str, str]:
        """Capture additional platform-specific information."""
        return {
            'platform': platform.platform(),
            'platform_node': platform.node(),
            'platform_machine': platform.machine(),
            'platform_processor': platform.processor(),
        }
    
    def _capture_git_info(self) -> Dict[str, str]:
        """Capture Git repository information."""
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            commit_hash = result.stdout.strip() if result.returncode == 0 else None
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            branch = result.stdout.strip() if result.returncode == 0 else None
            
            return {
                'commit_hash': commit_hash,
                'branch': branch
            }
        except Exception as e:
            print(f"Error capturing Git info: {e}")
            return {}
    
    def _set_random_seeds(self, experiment_id: str) -> Dict[str, int]:
        """Set random seeds for reproducibility based on experiment ID."""
        import random
        import numpy as np
        
        # Create a hash-based seed from the experiment ID
        seed_base = int(hashlib.md5(experiment_id.encode()).hexdigest()[:8], 16) % (2**32)
        
        # Set seeds for different libraries
        random.seed(seed_base)
        np.random.seed(seed_base)
        
        # For PyTorch (if available)
        try:
            import torch
            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_base)
                torch.cuda.manual_seed_all(seed_base)  # for multi-GPU
            # Ensure deterministic behavior in PyTorch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            print("PyTorch not available, skipping PyTorch seed setting")
        
        return {
            'python_random': seed_base,
            'numpy': seed_base,
            'pytorch': seed_base if 'torch' in sys.modules else None
        }
    
    def save_artifact(self, artifact: Any, name: str, format: str = "pickle") -> str:
        """Save an artifact (model, data, etc.) with a unique identifier."""
        if not self.experiment_running:
            raise RuntimeError("No active experiment. Call start_experiment first.")
        
        exp_dir = self.output_dir / self.metadata.experiment_id
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "pickle":
            filename = f"{name}_{timestamp}.pkl"
            filepath = artifacts_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(artifact, f)
        elif format == "json":
            filename = f"{name}_{timestamp}.json"
            filepath = artifacts_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Add to metadata artifacts list
        if self.metadata.artifacts is None:
            self.metadata.artifacts = []
        self.metadata.artifacts.append(str(filepath))
        
        print(f"Saved artifact: {filepath}")
        return str(filepath)
    
    @contextmanager
    def experiment_context(
        self,
        experiment_id: str,
        experiment_name: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Context manager for running an experiment with reproducibility tracking."""
        self.start_experiment(experiment_id, experiment_name, description, author, parameters)
        try:
            yield self
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            raise
        finally:
            self.end_experiment()


class ExperimentReplayer:
    """Replays experiments using saved metadata and artifacts."""
    
    def __init__(self, reproducibility_dir: str = "research/reproducibility"):
        self.reproducibility_dir = Path(reproducibility_dir)
    
    def load_experiment_metadata(self, experiment_id: str) -> ReproducibilityMetadata:
        """Load experiment metadata from saved file."""
        exp_dir = self.reproducibility_dir / experiment_id
        metadata_path = exp_dir / "reproducibility_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Convert dictionary back to ReproducibilityMetadata
        metadata = ReproducibilityMetadata(
            experiment_id=metadata_dict['experiment_id'],
            experiment_name=metadata_dict['experiment_name'],
            start_time=datetime.fromisoformat(metadata_dict['start_time']),
            end_time=datetime.fromisoformat(metadata_dict['end_time']) if metadata_dict['end_time'] else None,
            description=metadata_dict['description'],
            author=metadata_dict['author'],
            hardware_config=metadata_dict['hardware_config'],
            software_environment=metadata_dict['software_environment'],
            dependencies=metadata_dict['dependencies'],
            random_seeds=metadata_dict['random_seeds'],
            parameters=metadata_dict['parameters'],
            results=metadata_dict['results'],
            artifacts=metadata_dict['artifacts'],
            git_commit_hash=metadata_dict['git_commit_hash'],
            git_branch=metadata_dict['git_branch'],
            platform_info=metadata_dict['platform_info']
        )
        
        return metadata
    
    def verify_environment_match(self, required_metadata: ReproducibilityMetadata) -> Dict[str, Any]:
        """Verify if the current environment matches the requirements for reproducing an experiment."""
        current_software_env = self._capture_current_software_environment()
        current_hardware_config = self._capture_current_hardware_config()
        
        # Check software environment
        software_match = self._compare_software_env(
            required_metadata.software_environment, 
            current_software_env
        )
        
        # Check hardware configuration
        hardware_match = self._compare_hardware_config(
            required_metadata.hardware_config,
            current_hardware_config
        )
        
        # Check dependencies
        dependency_match = self._compare_dependencies(
            required_metadata.dependencies
        )
        
        return {
            'software_match': software_match,
            'hardware_match': hardware_match,
            'dependency_match': dependency_match,
            'overall_compatibility': software_match['match'] and hardware_match['match'] and dependency_match['match']
        }
    
    def _capture_current_software_environment(self) -> Dict[str, Any]:
        """Capture the current software environment."""
        return {
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'os_name': os.name,
            'platform_system': platform.system(),
        }
    
    def _capture_current_hardware_config(self) -> Dict[str, Any]:
        """Capture the current hardware configuration."""
        return {
            'processor': platform.processor(),
            'machine': platform.machine(),
        }
    
    def _compare_software_env(self, required: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare required and current software environments."""
        match = True
        mismatches = []
        
        # Check Python version (major.minor should match)
        if required.get('python_version', '').split('.')[:2] != current.get('python_version', '').split('.')[:2]:
            mismatches.append(
                f"Python version mismatch: required {required.get('python_version')}, "
                f"current {current.get('python_version')}"
            )
            match = False
        
        # Check OS
        if required.get('platform_system', '').lower() != current.get('platform_system', '').lower():
            mismatches.append(
                f"OS mismatch: required {required.get('platform_system')}, "
                f"current {current.get('platform_system')}"
            )
            match = False
        
        return {
            'match': match,
            'mismatches': mismatches,
            'required': required,
            'current': current
        }
    
    def _compare_hardware_config(self, required: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare required and current hardware configurations."""
        match = True
        mismatches = []
        
        # For now, just check if the processor architecture is similar
        # In a real implementation, this would have more sophisticated checks
        required_proc = required.get('processor', '').lower()
        current_proc = current.get('processor', '').lower()
        
        if required_proc and current_proc and not any(arch in current_proc for arch in ['x86', 'amd', 'intel'] if arch in required_proc):
            mismatches.append(
                f"Processor architecture mismatch: required {required_proc}, current {current_proc}"
            )
            match = False
        
        return {
            'match': match,
            'mismatches': mismatches,
            'required': required,
            'current': current
        }
    
    def _compare_dependencies(self, required_deps: Dict[str, str]) -> Dict[str, Any]:
        """Compare required and current Python dependencies."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format', 'freeze'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    'match': False,
                    'mismatches': [f"Could not list current dependencies: {result.stderr}"]
                }
            
            current_deps = {}
            for line in result.stdout.strip().split('\n'):
                if line and '==' in line:
                    name, version = line.split('==', 1)
                    current_deps[name] = version
            
            mismatches = []
            match = True
            
            for pkg, required_ver in required_deps.items():
                current_ver = current_deps.get(pkg)
                if not current_ver:
                    mismatches.append(f"Missing required package: {pkg}=={required_ver}")
                    match = False
                elif current_ver != required_ver:
                    mismatches.append(f"Version mismatch for {pkg}: required {required_ver}, current {current_ver}")
                    match = False
            
            return {
                'match': match,
                'mismatches': mismatches,
                'required': required_deps,
                'current': current_deps
            }
        except Exception as e:
            return {
                'match': False,
                'mismatches': [f"Error checking dependencies: {e}"]
            }
    
    def replay_experiment(self, experiment_id: str, 
                         custom_parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Replay an experiment using saved metadata and artifacts."""
        try:
            metadata = self.load_experiment_metadata(experiment_id)
            print(f"Replaying experiment: {metadata.experiment_name} ({metadata.experiment_id})")
            
            # Verify environment compatibility
            env_check = self.verify_environment_match(metadata)
            if not env_check['overall_compatibility']:
                print("Environment compatibility check failed:")
                if not env_check['software_match']['match']:
                    for mismatch in env_check['software_match']['mismatches']:
                        print(f"  - {mismatch}")
                if not env_check['hardware_match']['match']:
                    for mismatch in env_check['hardware_match']['mismatches']:
                        print(f"  - {mismatch}")
                if not env_check['dependency_match']['match']:
                    for mismatch in env_check['dependency_match']['mismatches']:
                        print(f"  - {mismatch}")
                
                print("WARNING: Environment mismatch detected. Results may vary.")
            
            # Set the same random seeds used in the original experiment
            self._restore_random_seeds(metadata.random_seeds)
            
            # Apply the original parameters (or custom ones if provided)
            params = custom_parameters if custom_parameters is not None else metadata.parameters
            print(f"Using parameters: {params}")
            
            # In a real implementation, this would restore the code to the exact state
            # it was in during the original experiment and run it with the same inputs
            print("Replaying experiment logic (simplified for this example)...")
            
            # Log that this is a replay
            replay_log = {
                'replay_time': datetime.now().isoformat(),
                'original_experiment_id': metadata.experiment_id,
                'original_run_time': metadata.start_time.isoformat(),
                'replay_parameters': params
            }
            
            exp_dir = self.reproducibility_dir / experiment_id
            replay_log_path = exp_dir / f"replay_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(replay_log_path, 'w') as f:
                json.dump(replay_log, f, indent=2)
            
            print(f"Replay completed. Log saved to: {replay_log_path}")
            return True
            
        except Exception as e:
            print(f"Error replaying experiment: {e}")
            return False
    
    def _restore_random_seeds(self, seeds: Dict[str, int]):
        """Restore the random seeds from an experiment."""
        if not seeds:
            return
        
        import random
        if 'python_random' in seeds and seeds['python_random'] is not None:
            random.seed(seeds['python_random'])
        
        import numpy as np
        if 'numpy' in seeds and seeds['numpy'] is not None:
            np.random.seed(seeds['numpy'])
        
        try:
            import torch
            if 'pytorch' in seeds and seeds['pytorch'] is not None:
                torch.manual_seed(seeds['pytorch'])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seeds['pytorch'])
                    torch.cuda.manual_seed_all(seeds['pytorch'])
        except ImportError:
            pass  # PyTorch not available


def create_reproducible_function(func: Callable) -> Callable:
    """Decorator to make a function reproducible by capturing its parameters and results."""
    def wrapper(*args, experiment_id: str = None, experiment_name: str = None, **kwargs):
        if not experiment_id or not experiment_name:
            # If no experiment info provided, just run the function normally
            return func(*args, **kwargs)
        
        # Initialize reproducibility manager
        repro_manager = ReproducibilityManager()
        
        # Start experiment with function parameters as metadata
        params = {
            'args': args,
            'kwargs': kwargs,
            'function_name': func.__name__
        }
        
        repro_manager.start_experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=f"Reproducible run of function: {func.__name__}",
            parameters=params
        )
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Save result as an artifact
            repro_manager.save_artifact(result, f"function_result_{func.__name__}", format="pickle")
            
            # End experiment with results
            repro_manager.end_experiment(
                results={"function_output": f"Result saved as artifact"},
                artifacts=[f"function_result_{func.__name__}"]
            )
            
            return result
        except Exception as e:
            # End experiment with error information
            repro_manager.end_experiment(
                results={"error": str(e)},
                artifacts=[]
            )
            raise e
    
    return wrapper


def main():
    """Example usage of reproducibility tools."""
    print("=== Reproducibility Tools Demo ===")
    
    # Example 1: Using the ReproducibilityManager directly
    repro_manager = ReproducibilityManager()
    
    with repro_manager.experiment_context(
        experiment_id="demo_exp_001",
        experiment_name="Sample Robotics Experiment",
        description="Testing robot navigation with different parameters",
        parameters={"speed": 0.5, "algorithm": "dwa", "map": "simple_room"}
    ) as exp:
        # Simulate running an experiment
        print("Running actual experiment logic...")
        import time
        time.sleep(1)  # Simulate computation time
        
        # Simulate getting some results
        results = {
            "success_rate": 0.85,
            "avg_completion_time": 24.5,
            "collisions": 2
        }
        
        # Simulate saving an artifact (like a trained model or processed data)
        sample_artifact = {"model_weights": [0.1, 0.2, 0.3], "accuracy": 0.92}
        artifact_path = exp.save_artifact(sample_artifact, "navigation_model")
    
    print("\n=== Experiment Replayer Demo ===")
    
    # Example 2: Using the ExperimentReplayer
    replayer = ExperimentReplayer()
    
    # Verify if the current environment is compatible with the experiment
    env_check = replayer.verify_environment_match(
        replayer.load_experiment_metadata("demo_exp_001")
    )
    print(f"Environment compatibility: {env_check['overall_compatibility']}")
    
    # Replay the experiment
    replay_success = replayer.replay_experiment("demo_exp_001")
    print(f"Replay successful: {replay_success}")
    
    print("\n=== Reproducible Function Demo ===")
    
    # Example 3: Using the reproducible function decorator
    @create_reproducible_function
    def sample_robotics_algorithm(speed: float, algorithm: str, map_name: str):
        """A sample robotics algorithm that we want to make reproducible."""
        import random
        # Some computation that uses randomness
        result = {
            "path_length": random.uniform(5.0, 10.0),
            "success": random.random() > 0.2,  # 80% success rate
            "computation_time": random.uniform(0.5, 2.0)
        }
        return result
    
    # Run the reproducible function
    result = sample_robotics_algorithm(
        0.5, "dwa", "simple_room",
        experiment_id="demo_func_exp_001",
        experiment_name="Sample Algorithm Test"
    )
    
    print(f"Function result: {result}")


if __name__ == "__main__":
    main()