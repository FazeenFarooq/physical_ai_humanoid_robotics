"""
NVIDIA Isaac Deployment Tools

This module provides tools and utilities for deploying robotics applications
using NVIDIA Isaac SDK on various platforms, particularly the Jetson Orin.
"""

import os
import subprocess
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsaacDeploymentManager:
    """
    Manages deployment of robotics applications using NVIDIA Isaac SDK
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Isaac deployment manager
        
        Args:
            config_path: Path to deployment configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.deployment_root = Path(self.config.get('deployment_root', './deployment'))
        self.deployment_root.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load deployment configuration from file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def setup_deployment_environment(self, target_platform: str = "jetson-orin"):
        """
        Set up the deployment environment on the target platform
        
        Args:
            target_platform: Target platform (e.g., 'jetson-orin', 'desktop-cuda')
        """
        logger.info(f"Setting up deployment environment for {target_platform}")
        
        if target_platform == "jetson-orin":
            self._setup_jetson_environment()
        elif target_platform.startswith("isaac-sim"):
            self._setup_isaac_sim_environment()
        else:
            logger.warning(f"Unknown platform: {target_platform}, proceeding with generic setup")
    
    def _setup_jetson_environment(self):
        """
        Set up deployment environment for Jetson Orin
        """
        # Check if we're on a Jetson device
        jetson_board = self._get_jetson_board()
        if not jetson_board:
            logger.warning("Not running on a Jetson device, proceeding anyway")
        
        # Install required Isaac packages
        self._install_isaac_packages()
        
        # Configure system parameters for optimal performance
        self._configure_jetson_performance()
        
    def _get_jetson_board(self) -> Optional[str]:
        """
        Detect Jetson board type
        """
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().replace('\x00', '')
                logger.info(f"Detected Jetson board: {model}")
                return model
        except FileNotFoundError:
            logger.debug("Not running on a Jetson device")
            return None
    
    def _install_isaac_packages(self):
        """
        Install required Isaac packages
        """
        logger.info("Installing required Isaac packages...")
        
        # Install Isaac ROS packages if available
        packages = [
            "isaac_ros_common",
            "isaac_ros_detectnet",
            "isaac_ros_image_pipeline",
            "isaac_ros_visual_slam",
            "isaac_ros_bmi",
            "isaac_ros_nitros",
        ]
        
        for package in packages:
            try:
                # Check if package is available
                result = subprocess.run([
                    "apt", "list", "--installed", f"ros-humble-{package}"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.info(f"Installing {package}...")
                    subprocess.run([
                        "sudo", "apt", "install", "-y", f"ros-humble-{package}"
                    ], check=True)
                else:
                    logger.info(f"{package} already installed")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")
    
    def _configure_jetson_performance(self):
        """
        Configure Jetson for optimal performance in robotic applications
        """
        logger.info("Configuring Jetson performance settings...")
        
        # Set power mode to MAXN if available
        try:
            subprocess.run(["sudo", "nvpmodel", "-m", "0"], check=True)
            logger.info("Set Jetson power mode to MAXN")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvpmodel not available, skipping power mode configuration")
        
        # Enable fan control if available
        try:
            subprocess.run(["sudo", "jc", "-m", "2"], check=True)
            logger.info("Enabled Jetson fan control")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Jetson control tool not available, skipping fan configuration")
    
    def _setup_isaac_sim_environment(self):
        """
        Set up deployment environment for Isaac Sim
        """
        logger.info("Setting up Isaac Sim environment...")
        
        # Check for Isaac Sim installation
        isaac_sim_path = os.environ.get('ISAAC_SIM_PATH', '/opt/isaac-sim')
        if not Path(isaac_sim_path).exists():
            logger.warning(f"Isaac Sim not found at {isaac_sim_path}")
            return
        
        # Set up environment variables for Isaac Sim
        os.environ['ISAACSIM_PATH'] = isaac_sim_path
        
        logger.info("Isaac Sim environment ready")
    
    def package_application(self, 
                           source_path: str, 
                           output_path: str, 
                           platform: str = "jetson-orin") -> str:
        """
        Package application for deployment on specified platform
        
        Args:
            source_path: Path to source application
            output_path: Output path for packaged application
            platform: Target platform for deployment
            
        Returns:
            Path to the packaged application
        """
        logger.info(f"Packing application from {source_path} for {platform}")
        
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if platform == "jetson-orin":
            return self._package_for_jetson(source, output)
        elif platform.startswith("isaac-sim"):
            return self._package_for_isaac_sim(source, output)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _package_for_jetson(self, source: Path, output: Path) -> str:
        """
        Package application for Jetson deployment
        """
        logger.info(f"Packaging for Jetson: {source} -> {output}")
        
        # Create a tarball of the application
        import tarfile
        
        with tarfile.open(output, "w:gz") as tar:
            tar.add(source, arcname=source.name)
        
        logger.info(f"Application packaged successfully: {output}")
        return str(output)
    
    def _package_for_isaac_sim(self, source: Path, output: Path) -> str:
        """
        Package application for Isaac Sim deployment
        """
        logger.info(f"Packaging for Isaac Sim: {source} -> {output}")
        
        # For Isaac Sim, we typically just copy the files as-is
        import shutil
        
        if source.is_file():
            shutil.copy2(source, output)
        else:
            if output.exists() and output.is_dir():
                shutil.rmtree(output)
            shutil.copytree(source, output)
        
        logger.info(f"Application packaged successfully: {output}")
        return str(output)
    
    def deploy_to_target(self, 
                         package_path: str, 
                         target_address: str, 
                         target_path: str = "/home/deploy/app") -> bool:
        """
        Deploy packaged application to target device
        
        Args:
            package_path: Path to the packaged application
            target_address: Target device address (IP or hostname)
            target_path: Installation path on target device
            
        Returns:
            True if deployment was successful, False otherwise
        """
        logger.info(f"Deploying to {target_address} at {target_path}")
        
        package = Path(package_path)
        if not package.exists():
            logger.error(f"Package does not exist: {package_path}")
            return False
        
        # Use rsync or scp to transfer the package
        try:
            # First, create the target directory
            subprocess.run([
                "ssh", target_address, 
                f"mkdir -p {target_path}"
            ], check=True)
            
            # Transfer the package
            subprocess.run([
                "rsync", "-avz", 
                str(package), 
                f"{target_address}:{target_path}/"
            ], check=True)
            
            # Extract and set up the application (if it's a compressed package)
            if str(package).endswith('.tar.gz'):
                package_name = package.name
                subprocess.run([
                    "ssh", target_address,
                    f"cd {target_path} && tar -xzf {package_name} && rm {package_name}"
                ], check=True)
            
            logger.info(f"Successfully deployed to {target_address}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def deploy_isaac_application(self,
                                 app_config: Dict[str, Any],
                                 target_platform: str,
                                 target_address: str = None) -> bool:
        """
        Deploy an Isaac application to the specified platform
        
        Args:
            app_config: Configuration for the Isaac application
            target_platform: Target platform ('jetson-orin', 'isaac-sim', etc.)
            target_address: Target device address for remote deployment
            
        Returns:
            True if deployment was successful, False otherwise
        """
        logger.info(f"Deploying Isaac application to {target_platform}")
        
        # Validate application configuration
        if not self._validate_app_config(app_config):
            logger.error("Invalid application configuration")
            return False
        
        # Set up deployment environment
        self.setup_deployment_environment(target_platform)
        
        # Create deployment package
        app_source = app_config.get('source_path', '.')
        package_name = f"isaac_app_{app_config.get('name', 'default')}.tar.gz"
        package_path = self.deployment_root / package_name
        
        self.package_application(app_source, package_path, target_platform)
        
        # Deploy to target if specified
        if target_address:
            target_path = app_config.get('target_path', '/home/deploy/isaac_app')
            return self.deploy_to_target(package_path, target_address, target_path)
        
        logger.info(f"Application packaged at: {package_path}")
        return True
    
    def _validate_app_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the application configuration
        """
        required_keys = ['name', 'source_path']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        return True
    
    def monitor_deployment(self, 
                          target_address: str, 
                          app_name: str) -> Dict[str, Any]:
        """
        Monitor the deployed application on the target device
        
        Args:
            target_address: Target device address
            app_name: Name of the deployed application
            
        Returns:
            Dictionary with monitoring information
        """
        logger.info(f"Monitoring application {app_name} on {target_address}")
        
        # Check if the application is running
        try:
            result = subprocess.run([
                "ssh", target_address,
                f"pgrep -f {app_name} || echo 'not_running'"
            ], capture_output=True, text=True, check=True)
            
            is_running = result.stdout.strip() != "not_running"
            
            # Get resource usage if the application is running
            if is_running:
                pid = result.stdout.strip()
                # Get CPU and memory usage
                usage_result = subprocess.run([
                    "ssh", target_address,
                    f"ps -p {pid} -o pid,pcpu,pmem,comm --no-headers"
                ], capture_output=True, text=True, check=True)
                
                if usage_result.returncode == 0:
                    parts = usage_result.stdout.strip().split()
                    if len(parts) >= 4:
                        return {
                            "running": True,
                            "pid": pid,
                            "cpu_percent": float(parts[1]),
                            "mem_percent": float(parts[2]),
                            "command": parts[3]
                        }
            
            return {
                "running": False,
                "pid": None,
                "cpu_percent": 0,
                "mem_percent": 0,
                "command": None
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to monitor application: {e}")
            return {
                "running": False,
                "error": str(e)
            }


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="NVIDIA Isaac Deployment Tools")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--platform", type=str, default="jetson-orin", 
                       help="Target platform for deployment")
    parser.add_argument("--target", type=str, help="Target device address for deployment")
    parser.add_argument("--source", type=str, help="Source application path")
    parser.add_argument("--deploy", action="store_true", help="Deploy the application")
    
    args = parser.parse_args()
    
    manager = IsaacDeploymentManager(args.config)
    
    if args.deploy and args.source:
        app_config = {
            'name': 'default_app',
            'source_path': args.source
        }
        success = manager.deploy_isaac_application(
            app_config, 
            args.platform, 
            args.target
        )
        if success:
            print("Deployment successful!")
            sys.exit(0)
        else:
            print("Deployment failed!")
            sys.exit(1)
    else:
        print("Isaac Deployment Manager initialized.")
        manager.setup_deployment_environment(args.platform)


if __name__ == "__main__":
    main()