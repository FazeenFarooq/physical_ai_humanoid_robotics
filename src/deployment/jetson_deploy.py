"""
Jetson Orin Deployment Tools for the Physical AI & Humanoid Robotics Course.

This module provides tools for deploying AI models and robotics applications to NVIDIA Jetson Orin hardware,
following the course's emphasis on GPU-accelerated perception and training.
"""

import os
import subprocess
import sys
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tarfile
import tempfile
import logging
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration for Jetson Orin deployment."""
    target_ip: str
    target_user: str = "jetson"
    target_port: int = 22
    ssh_key_path: Optional[str] = None
    model_path: str = ""
    app_name: str = "robot_app"
    workspace_dir: str = "/home/jetson/robot_workspace"
    build_dir: str = "build"
    install_dir: str = "install"
    dependencies: List[str] = None
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.env_vars is None:
            self.env_vars = {}


class JetsonDeployer:
    """Tool for deploying robotics applications to Jetson Orin."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the deployer."""
        logger = logging.getLogger(f"JetsonDeployer-{self.config.target_ip}")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def check_connection(self) -> bool:
        """Check if we can connect to the Jetson device."""
        try:
            # Test SSH connection
            result = subprocess.run([
                "ssh", 
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                "echo 'Connection successful'"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                self.logger.info("Successfully connected to Jetson Orin")
                return True
            else:
                self.logger.error(f"Failed to connect to Jetson: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Connection timed out")
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def prepare_package(self, source_dir: str, package_path: str) -> bool:
        """Prepare a deployment package from source directory."""
        try:
            # Create temporary directory for packaging
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy source files
                src_path = Path(source_dir)
                dest_path = temp_path / "app"
                shutil.copytree(src_path, dest_path)
                
                # Create deployment metadata
                metadata = {
                    "app_name": self.config.app_name,
                    "deploy_timestamp": str(Path(source_dir).stat().st_mtime),
                    "target_architecture": "aarch64",
                    "nvidia_platform": "jetson-orin"
                }
                
                with open(temp_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Create package archive
                with tarfile.open(package_path, "w:gz") as tar:
                    tar.add(temp_path, arcname=".")
                
                self.logger.info(f"Package created at {package_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to prepare package: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies on the Jetson device."""
        if not self.config.dependencies:
            self.logger.info("No dependencies to install")
            return True
        
        try:
            # Create command to install dependencies
            deps_str = " ".join(self.config.dependencies)
            cmd = f"sudo apt update && sudo apt install -y {deps_str}"
            
            result = subprocess.run([
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                cmd
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                self.logger.info("Dependencies installed successfully")
                return True
            else:
                self.logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Dependency installation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Dependency installation error: {e}")
            return False
    
    def upload_package(self, package_path: str) -> bool:
        """Upload the deployment package to the Jetson device."""
        try:
            # Upload package to Jetson
            result = subprocess.run([
                "scp",
                "-o", "StrictHostKeyChecking=no",
                package_path,
                f"{self.config.target_user}@{self.config.target_ip}:{self.config.workspace_dir}/"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout for large packages
            
            if result.returncode == 0:
                self.logger.info(f"Package uploaded to Jetson: {package_path}")
                return True
            else:
                self.logger.error(f"Failed to upload package: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Package upload timed out")
            return False
        except Exception as e:
            self.logger.error(f"Package upload error: {e}")
            return False
    
    def extract_and_deploy(self, package_filename: str) -> bool:
        """Extract and deploy the application on the Jetson device."""
        try:
            # Extract and set up the application
            extract_cmd = f"""
            cd {self.config.workspace_dir} &&
            rm -rf {self.config.app_name} &&
            mkdir -p {self.config.app_name} &&
            cd {self.config.app_name} &&
            tar -xzf ../{package_filename} &&
            cd app &&
            mkdir -p {self.config.build_dir} &&
            cd {self.config.build_dir} &&
            cmake .. &&
            make -j$(nproc)
            """
            
            result = subprocess.run([
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                extract_cmd
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                self.logger.info(f"Application deployed to Jetson: {self.config.app_name}")
                return True
            else:
                self.logger.error(f"Failed to deploy application: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Application extraction and deployment timed out")
            return False
        except Exception as e:
            self.logger.error(f"Application deployment error: {e}")
            return False
    
    def configure_environment(self) -> bool:
        """Configure the runtime environment on the Jetson device."""
        try:
            # Set up environment variables and configurations
            env_setup = f"""
            cd {self.config.workspace_dir}/{self.config.app_name}/app &&
            echo '#!/bin/bash' > setup_env.sh &&
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> setup_env.sh &&
            echo 'export CUDA_HOME=/usr/local/cuda' >> setup_env.sh
            """
            
            for key, value in self.config.env_vars.items():
                env_setup += f"echo 'export {key}={value}' >> setup_env.sh &&"
            
            # Make executable
            env_setup += "chmod +x setup_env.sh"
            
            result = subprocess.run([
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                env_setup
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("Environment configured on Jetson")
                return True
            else:
                self.logger.error(f"Failed to configure environment: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Environment configuration error: {e}")
            return False
    
    def start_application(self) -> bool:
        """Start the deployed application."""
        try:
            # Start the application
            start_cmd = f"""
            cd {self.config.workspace_dir}/{self.config.app_name}/app &&
            source setup_env.sh &&
            ./{self.config.build_dir}/{self.config.app_name} &
            """
            
            result = subprocess.run([
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                start_cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.logger.info(f"Application started on Jetson: {self.config.app_name}")
                return True
            else:
                self.logger.error(f"Failed to start application: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Application start error: {e}")
            return False
    
    def deploy(self, source_dir: str) -> bool:
        """Full deployment process."""
        self.logger.info(f"Starting deployment to Jetson Orin at {self.config.target_ip}")
        
        # 1. Check connection
        if not self.check_connection():
            return False
        
        # 2. Install dependencies
        if not self.install_dependencies():
            self.logger.error("Failed to install dependencies")
            return False
        
        # 3. Configure environment
        if not self.configure_environment():
            self.logger.error("Failed to configure environment")
            return False
        
        # 4. Prepare package
        package_name = f"{self.config.app_name}_package.tar.gz"
        if not self.prepare_package(source_dir, package_name):
            self.logger.error("Failed to prepare package")
            return False
        
        # 5. Upload package
        if not self.upload_package(package_name):
            self.logger.error("Failed to upload package")
            return False
        
        # 6. Extract and deploy
        if not self.extract_and_deploy(package_name):
            self.logger.error("Failed to extract and deploy package")
            return False
        
        # 7. Start application
        if not self.start_application():
            self.logger.error("Failed to start application")
            return False
        
        self.logger.info("Deployment completed successfully")
        return True
    
    def monitor_deployment(self) -> Dict[str, Any]:
        """Monitor the deployed application."""
        try:
            monitor_cmd = f"""
            echo "CPU Usage:" &&
            top -bn1 | grep "Cpu(s)" &&
            echo "Memory Usage:" &&
            free -m &&
            echo "GPU Status:" &&
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv &&
            echo "Process Status:" &&
            ps aux | grep {self.config.app_name}
            """
            
            result = subprocess.run([
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.target_user}@{self.config.target_ip}",
                monitor_cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "monitor_data": result.stdout
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cleanup(self, keep_package: bool = False) -> bool:
        """Clean up temporary files after deployment."""
        try:
            # Remove temporary package file
            if not keep_package:
                package_name = f"{self.config.app_name}_package.tar.gz"
                if os.path.exists(package_name):
                    os.remove(package_name)
                    self.logger.info(f"Removed temporary package: {package_name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            return False


def create_deployment_config_from_dict(config_dict: Dict[str, Any]) -> DeploymentConfig:
    """Create a DeploymentConfig from a dictionary."""
    return DeploymentConfig(
        target_ip=config_dict.get("target_ip", "localhost"),
        target_user=config_dict.get("target_user", "jetson"),
        target_port=config_dict.get("target_port", 22),
        ssh_key_path=config_dict.get("ssh_key_path"),
        model_path=config_dict.get("model_path", ""),
        app_name=config_dict.get("app_name", "robot_app"),
        workspace_dir=config_dict.get("workspace_dir", "/home/jetson/robot_workspace"),
        build_dir=config_dict.get("build_dir", "build"),
        install_dir=config_dict.get("install_dir", "install"),
        dependencies=config_dict.get("dependencies", []),
        env_vars=config_dict.get("env_vars", {})
    )


def example_deployment():
    """Example of how to use the Jetson deployment tools."""
    # Example configuration
    config = DeploymentConfig(
        target_ip="192.168.1.20",  # Jetson Orin IP address
        app_name="perception_pipeline",
        dependencies=["ros-humble-navigation2", "ros-humble-vision-opencv"],
        env_vars={
            "ROS_DOMAIN_ID": "1",
            "CUDA_CACHE_PATH": "/home/jetson/.cuda_cache"
        }
    )
    
    # Initialize deployer
    deployer = JetsonDeployer(config)
    
    # Perform deployment
    success = deployer.deploy("/path/to/robot/application")
    
    if success:
        print("Deployment successful!")
        
        # Monitor the deployment
        monitor_result = deployer.monitor_deployment()
        print("Deployment monitoring result:", monitor_result)
    else:
        print("Deployment failed!")
    
    # Clean up
    deployer.cleanup()


if __name__ == "__main__":
    example_deployment()