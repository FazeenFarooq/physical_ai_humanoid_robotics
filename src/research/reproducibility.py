"""
Reproducibility Tools for Research Validation

This module provides tools to ensure that experiments in the Physical AI & 
Humanoid Robotics course are reproducible across different systems and 
environments.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import platform
import psutil
import git
from dataclasses import dataclass


@dataclass
class SystemFingerprint:
    """Data class to hold system fingerprint information"""
    hardware_id: str
    os_info: str
    python_version: str
    ros_version: str
    cuda_version: str
    pytorch_version: str
    git_commit_hash: str
    timestamp: datetime
    environment_variables: Dict[str, str]


class ReproducibilityManager:
    """
    Manages reproducibility of experiments by capturing system state and 
    ensuring consistent environments across different systems.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the reproducibility manager
        
        Args:
            experiment_dir: Directory for the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.system_fingerprint_path = self.experiment_dir / "system_fingerprint.json"
        self.environment_snapshot_path = self.experiment_dir / "environment_snapshot.json"
        self.software_manifest_path = self.experiment_dir / "software_manifest.json"
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_system_fingerprint(self) -> SystemFingerprint:
        """
        Capture a comprehensive system fingerprint
        
        Returns:
            SystemFingerprint object containing system information
        """
        # Get hardware ID (CPU info, memory, etc.)
        hardware_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('.').total,
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        hardware_id = hashlib.sha256(json.dumps(hardware_info, sort_keys=True).encode()).hexdigest()
        
        # Get OS information
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.architecture()
        }
        
        # Get Python version
        python_version = platform.python_version()
        
        # Get ROS version (try different methods)
        ros_version = self._get_ros_version()
        
        # Get CUDA version
        cuda_version = self._get_cuda_version()
        
        # Get PyTorch version
        pytorch_version = self._get_pytorch_version()
        
        # Get current git commit hash
        git_commit_hash = self._get_git_commit_hash()
        
        # Get environment variables (filter for relevant ones)
        env_vars = {
            k: v for k, v in os.environ.items() 
            if any(keyword in k.lower() for keyword in 
                   ['ros', 'cuda', 'torch', 'python', 'path', 'home', 'user', 'lang'])
        }
        
        # Create system fingerprint
        fingerprint = SystemFingerprint(
            hardware_id=hardware_id,
            os_info=json.dumps(os_info),
            python_version=python_version,
            ros_version=ros_version,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
            git_commit_hash=git_commit_hash,
            timestamp=datetime.now(),
            environment_variables=env_vars
        )
        
        return fingerprint
    
    def _get_ros_version(self) -> str:
        """Get ROS version information"""
        try:
            # Try to get ROS_DISTRO environment variable
            ros_distro = os.environ.get('ROS_DISTRO', 'unknown')
            
            # Try to get ROS version from command line
            result = subprocess.run(['rosversion', '-d'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # For ROS 2, try different command
            result = subprocess.run(['ros2', '--version'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return f"ROS 2 ({result.stdout.strip()})"
                
            return ros_distro
        except Exception:
            return "unknown"
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version"""
        try:
            # Try nvidia-smi
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                # Extract CUDA version from the output
                import re
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if cuda_match:
                    return cuda_match.group(1)
            
            # Try nvcc
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
                
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_pytorch_version(self) -> str:
        """Get PyTorch version"""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "not installed"
    
    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except Exception:
            return "unknown"
    
    def save_system_fingerprint(self, fingerprint: SystemFingerprint) -> str:
        """
        Save the system fingerprint to a file
        
        Args:
            fingerprint: SystemFingerprint object to save
            
        Returns:
            Path to the saved file
        """
        fingerprint_dict = {
            "hardware_id": fingerprint.hardware_id,
            "os_info": json.loads(fingerprint.os_info),
            "python_version": fingerprint.python_version,
            "ros_version": fingerprint.ros_version,
            "cuda_version": fingerprint.cuda_version,
            "pytorch_version": fingerprint.pytorch_version,
            "git_commit_hash": fingerprint.git_commit_hash,
            "timestamp": fingerprint.timestamp.isoformat(),
            "environment_variables": fingerprint.environment_variables
        }
        
        with open(self.system_fingerprint_path, 'w') as f:
            json.dump(fingerprint_dict, f, indent=2, default=str)
        
        return str(self.system_fingerprint_path)
    
    def capture_environment_snapshot(self) -> str:
        """
        Capture the current environment state (packages, dependencies, etc.)
        
        Returns:
            Path to the saved environment snapshot
        """
        # Get Python packages
        pip_packages = self._get_pip_packages()
        
        # Get system environment variables
        system_env = dict(os.environ)
        
        # Get current working directory and repository info
        cwd = os.getcwd()
        try:
            repo = git.Repo(search_parent_directories=True)
            repo_info = {
                "path": repo.working_dir,
                "active_branch": repo.active_branch.name,
                "uncommitted_changes": len([item for item in repo.index.diff(None)]) > 0,
                "remotes": [remote.name for remote in repo.remotes]
            }
        except Exception:
            repo_info = {"error": "Not in a git repository or git error"}
        
        # Get system resource usage
        system_resources = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('.').percent
        }
        
        # Create environment snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "pip_packages": pip_packages,
            "system_environment": system_env,
            "repository_info": repo_info,
            "system_resources": system_resources,
            "python_paths": sys.path
        }
        
        with open(self.environment_snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        return str(self.environment_snapshot_path)
    
    def _get_pip_packages(self) -> List[Dict[str, str]]:
        """Get list of installed pip packages with their versions"""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format', 'json'], 
                                  capture_output=True, text=True, check=True)
            packages = json.loads(result.stdout)
            return packages
        except Exception as e:
            print(f"Error getting pip packages: {e}")
            # Fallback: manually get some common packages
            packages = []
            common_packages = [
                'torch', 'torchvision', 'torchaudio', 
                'numpy', 'pandas', 'matplotlib', 'opencv-python',
                'transformers', 'openai', 'ros-nodes'
            ]
            
            for pkg in common_packages:
                try:
                    __import__(pkg)
                    pkg_module = sys.modules[pkg]
                    version = getattr(pkg_module, '__version__', 'unknown')
                    packages.append({"name": pkg, "version": version})
                except ImportError:
                    continue
            
            return packages
    
    def create_software_manifest(self) -> str:
        """
        Create a software manifest of all relevant packages and their versions
        
        Returns:
            Path to the saved software manifest
        """
        # Get all the package information
        pip_packages = self._get_pip_packages()
        
        # Filter for relevant packages related to our project
        relevant_packages = []
        for pkg in pip_packages:
            if any(keyword in pkg['name'].lower() for keyword in 
                   ['torch', 'ros', 'gazebo', 'isaac', 'pytorch', 'cuda', 'cv2', 'open3d', 
                    'numpy', 'pandas', 'matplotlib', 'transformers', 'openai', 'nltk', 'spacy']):
                relevant_packages.append(pkg)
        
        # Add system-level information
        manifest = {
            "created_at": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture(),
            },
            "relevant_packages": relevant_packages,
            "pip_freeze_output": self._get_pip_freeze()
        }
        
        with open(self.software_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        return str(self.software_manifest_path)
    
    def _get_pip_freeze(self) -> str:
        """Get the output of pip freeze"""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout
        except Exception as e:
            return f"Error getting pip freeze: {e}"
    
    def validate_environment(self, reference_dir: str) -> Dict[str, Any]:
        """
        Validate if the current environment matches a reference environment
        
        Args:
            reference_dir: Directory containing the reference environment data
            
        Returns:
            Dictionary with validation results
        """
        reference_dir = Path(reference_dir)
        validation_results = {
            "environment_match": True,
            "differences": [],
            "warnings": [],
            "system_fingerprint_match": False,
            "software_manifest_match": False
        }
        
        # Load reference system fingerprint
        ref_fingerprint_path = reference_dir / "system_fingerprint.json"
        if ref_fingerprint_path.exists():
            with open(ref_fingerprint_path, 'r') as f:
                ref_fingerprint = json.load(f)
            
            # Compare with current fingerprint
            current_fingerprint = self.capture_system_fingerprint()
            current_fingerprint_dict = {
                "hardware_id": current_fingerprint.hardware_id,
                "os_info": json.loads(current_fingerprint.os_info),
                "python_version": current_fingerprint.python_version,
                "ros_version": current_fingerprint.ros_version,
                "cuda_version": current_fingerprint.cuda_version,
                "pytorch_version": current_fingerprint.pytorch_version,
                "git_commit_hash": current_fingerprint.git_commit_hash,
            }
            
            # Check git commit hash - this is likely to differ
            if current_fingerprint_dict["git_commit_hash"] != ref_fingerprint["git_commit_hash"]:
                validation_results["warnings"].append(
                    f"Git commit hash differs: current={current_fingerprint_dict['git_commit_hash'][:8]}, reference={ref_fingerprint['git_commit_hash'][:8]}"
                )
                # Remove from comparison since it's expected to differ
                del current_fingerprint_dict["git_commit_hash"]
                del ref_fingerprint["git_commit_hash"]
            
            # Compare fingerprints
            if current_fingerprint_dict == ref_fingerprint:
                validation_results["system_fingerprint_match"] = True
            else:
                validation_results["differences"].append("System fingerprint differs")
                validation_results["environment_match"] = False
        
        # Load reference software manifest
        ref_manifest_path = reference_dir / "software_manifest.json"
        if ref_manifest_path.exists():
            with open(ref_manifest_path, 'r') as f:
                ref_manifest = json.load(f)
            
            # Capture current software manifest
            current_manifest_path = self.create_software_manifest()
            with open(current_manifest_path, 'r') as f:
                current_manifest = json.load(f)
            
            # Compare relevant packages
            ref_packages = {pkg['name']: pkg['version'] for pkg in ref_manifest['relevant_packages']}
            current_packages = {pkg['name']: pkg['version'] for pkg in current_manifest['relevant_packages']}
            
            all_packages_match = True
            for pkg_name, ref_version in ref_packages.items():
                if pkg_name not in current_packages:
                    validation_results["differences"].append(f"Missing package: {pkg_name}")
                    all_packages_match = False
                elif current_packages[pkg_name] != ref_version:
                    validation_results["differences"].append(
                        f"Version mismatch for {pkg_name}: current={current_packages[pkg_name]}, reference={ref_version}"
                    )
                    all_packages_match = False
            
            # Check for extra packages in current environment
            for pkg_name in current_packages:
                if pkg_name not in ref_packages:
                    validation_results["warnings"].append(f"Extra package in current environment: {pkg_name}")
            
            if all_packages_match:
                validation_results["software_manifest_match"] = True
            else:
                validation_results["environment_match"] = False
        
        return validation_results
    
    def run(self) -> Dict[str, str]:
        """
        Run the complete reproducibility capture process
        
        Returns:
            Dictionary containing paths to all created files
        """
        # Capture system fingerprint
        fingerprint = self.capture_system_fingerprint()
        fingerprint_path = self.save_system_fingerprint(fingerprint)
        
        # Capture environment snapshot
        snapshot_path = self.capture_environment_snapshot()
        
        # Create software manifest
        manifest_path = self.create_software_manifest()
        
        return {
            "system_fingerprint": fingerprint_path,
            "environment_snapshot": snapshot_path,
            "software_manifest": manifest_path
        }


def create_docker_for_environment(experiment_dir: str, output_path: Optional[str] = None) -> str:
    """
    Creates a Dockerfile that can recreate the current environment
    
    Args:
        experiment_dir: Directory containing the experiment
        output_path: Path to save the Dockerfile (optional, defaults to experiment dir)
        
    Returns:
        Path to the created Dockerfile
    """
    rm = ReproducibilityManager(experiment_dir)
    
    # Load system information
    fingerprint_path = Path(experiment_dir) / "system_fingerprint.json"
    if fingerprint_path.exists():
        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)
    else:
        # If no fingerprint exists, create one
        fingerprint = rm.capture_system_fingerprint()
        rm.save_system_fingerprint(fingerprint)
        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)
    
    # Determine base image based on OS and CUDA
    cuda_version = fingerprint.get('cuda_version', 'unknown')
    if '12.' in str(cuda_version):
        base_image = f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"
    elif '11.' in str(cuda_version):
        base_image = f"nvidia/cuda:{cuda_version}-devel-ubuntu20.04"
    else:
        base_image = "ubuntu:22.04"
    
    # Get Python version
    python_version = fingerprint.get('python_version', '3.10')
    python_version_short = '.'.join(python_version.split('.')[:2])
    
    # Create Dockerfile content
    dockerfile_content = f"""# Generated Dockerfile for reproducible environment
# Based on system fingerprint captured at {fingerprint.get('timestamp', 'unknown')}

FROM {base_image}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO={fingerprint.get('ros_version', 'humble').split('(')[-1].replace(')', '') if 'ROS 2' in str(fingerprint.get('ros_version', '')) else fingerprint.get('ros_version', 'humble')}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    python3-setuptools \\
    build-essential \\
    git \\
    curl \\
    gnupg \\
    lsb-release \\
    && rm -rf /var/lib/apt/lists/* \\
    && ln -s /usr/bin/python3 /usr/bin/python

# Set up Python
RUN python -m pip install --upgrade pip

# Install ROS (if needed)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \\
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \\
    && apt-get update \\
    && apt-get install -y ros-${{ROS_DISTRO}}-desktop ros-${{ROS_DISTRO}}-vision-opencv \\
    && apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \\
    && rm -rf /var/lib/apt/lists/* \\
    && rosdep update

# Set up workspace
ENV ROS_PATH=/opt/ros/${{ROS_DISTRO}}
RUN echo "source $ROS_PATH/setup.bash" >> ~/.bashrc

# Create and set working directory
WORKDIR /workspace

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# If PyTorch is specified, install with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the rest of the application
COPY . .

# Source ROS environment
RUN echo "source $ROS_PATH/setup.bash" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash"]
"""
    
    # Determine output path
    if output_path:
        dockerfile_path = Path(output_path)
    else:
        dockerfile_path = Path(experiment_dir) / "Dockerfile"
    
    # Write Dockerfile
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    return str(dockerfile_path)


def validate_reproducibility(experiment_dir: str, reference_dir: str) -> Dict[str, Any]:
    """
    Validate reproducibility by comparing an experiment directory with a reference
    
    Args:
        experiment_dir: Directory of the experiment to validate
        reference_dir: Directory of the reference experiment
        
    Returns:
        Dictionary with validation results
    """
    rm = ReproducibilityManager(experiment_dir)
    return rm.validate_environment(reference_dir)


if __name__ == "__main__":
    # Example usage
    print("Creating reproducibility snapshot...")
    
    # Initialize manager
    rm = ReproducibilityManager("./sample_experiment")
    
    # Run full reproducibility capture
    paths = rm.run()
    
    print("Created files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Create Dockerfile for environment reproduction
    docker_path = create_docker_for_environment("./sample_experiment")
    print(f"\nCreated Dockerfile: {docker_path}")
    
    print("\nReproducibility manager initialized and executed successfully!")