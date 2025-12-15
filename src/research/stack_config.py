"""
Standardized hardware and software stack configuration for research validation.

This module provides tools for configuring and managing standardized hardware
and software stacks for reproducible research in Physical AI & Humanoid Robotics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import os
from pathlib import Path


@dataclass
class HardwareComponent:
    """Represents a hardware component in the standardized stack."""
    name: str
    model: str
    version: str
    specifications: Dict[str, str]
    dependencies: List[str]


@dataclass
class SoftwareComponent:
    """Represents a software component in the standardized stack."""
    name: str
    version: str
    dependencies: List[str]
    configuration_files: List[str]


@dataclass
class StackConfiguration:
    """Represents a complete hardware and software stack configuration."""
    name: str
    description: str
    hardware_components: List[HardwareComponent]
    software_components: List[SoftwareComponent]
    environment_variables: Dict[str, str]
    requirements_file: str


class StackConfigManager:
    """Manages standardized hardware and software stack configurations."""

    def __init__(self, config_dir: str = "configs/stacks"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, StackConfiguration] = {}
        self.load_all_configs()

    def load_config(self, config_name: str) -> StackConfiguration:
        """Load a stack configuration from YAML file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create HardwareComponent objects
        hardware_components = []
        for hw_data in data.get('hardware_components', []):
            hw_comp = HardwareComponent(
                name=hw_data['name'],
                model=hw_data['model'],
                version=hw_data['version'],
                specifications=hw_data.get('specifications', {}),
                dependencies=hw_data.get('dependencies', [])
            )
            hardware_components.append(hw_comp)
        
        # Create SoftwareComponent objects
        software_components = []
        for sw_data in data.get('software_components', []):
            sw_comp = SoftwareComponent(
                name=sw_data['name'],
                version=sw_data['version'],
                dependencies=sw_data.get('dependencies', []),
                configuration_files=sw_data.get('configuration_files', [])
            )
            software_components.append(sw_comp)
        
        config = StackConfiguration(
            name=data['name'],
            description=data['description'],
            hardware_components=hardware_components,
            software_components=software_components,
            environment_variables=data.get('environment_variables', {}),
            requirements_file=data.get('requirements_file', 'requirements.txt')
        )
        
        self.configs[config_name] = config
        return config
    
    def load_all_configs(self):
        """Load all available stack configurations."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Create a default configuration
            self.create_default_config()
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            try:
                self.load_config(config_name)
            except Exception as e:
                print(f"Error loading config {config_name}: {e}")
    
    def create_default_config(self):
        """Create a default research stack configuration."""
        default_config = {
            'name': 'research_default',
            'description': 'Default research stack for Physical AI experiments',
            'hardware_components': [
                {
                    'name': 'robot_platform',
                    'model': 'Unitree H1',
                    'version': '1.0',
                    'specifications': {
                        'height': '1.7m',
                        'weight': '47kg',
                        'degrees_of_freedom': '23',
                        'battery_life': '2.5h',
                        'computational_power': 'Jetson Orin AGX (275 TOPS)'
                    },
                    'dependencies': []
                },
                {
                    'name': 'development_workstation',
                    'model': 'RTX Workstation',
                    'version': '2025',
                    'specifications': {
                        'cpu': 'AMD Ryzen 7 7800X3D',
                        'gpu': 'NVIDIA RTX 4090',
                        'ram': '64GB DDR5',
                        'storage': '2TB NVMe SSD'
                    },
                    'dependencies': []
                }
            ],
            'software_components': [
                {
                    'name': 'ros_2',
                    'version': 'humble',
                    'dependencies': [],
                    'configuration_files': ['config/ros/ros2_config.yaml']
                },
                {
                    'name': 'pytorch',
                    'version': '2.0+',
                    'dependencies': [],
                    'configuration_files': ['config/ai/pytorch_config.yaml']
                },
                {
                    'name': 'isaac_sim',
                    'version': '4.0+',
                    'dependencies': [],
                    'configuration_files': ['config/simulation/isaac_config.yaml']
                }
            ],
            'environment_variables': {
                'ROS_DISTRO': 'humble',
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTHONPATH': '${PYTHONPATH}:src/'
            },
            'requirements_file': 'requirements.txt'
        }
        
        config_path = self.config_dir / "research_default.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def validate_stack(self, config_name: str) -> List[str]:
        """Validate that a stack configuration can be properly set up."""
        errors = []
        
        if config_name not in self.configs:
            errors.append(f"Configuration '{config_name}' not found")
            return errors
        
        config = self.configs[config_name]
        
        # Check software dependencies
        for sw_comp in config.software_components:
            # Check if requirements file exists
            req_path = Path(sw_comp.configuration_files[0] if sw_comp.configuration_files else config.requirements_file)
            if not req_path.exists():
                errors.append(f"Requirements file {req_path} not found for {sw_comp.name}")
        
        # Validate environment variables
        for var_name, var_value in config.environment_variables.items():
            if not var_name or not var_value:
                errors.append(f"Invalid environment variable: {var_name}={var_value}")
        
        return errors
    
    def setup_stack(self, config_name: str) -> bool:
        """Set up the hardware and software stack as specified in the configuration."""
        if config_name not in self.configs:
            print(f"Configuration '{config_name}' not found")
            return False
        
        config = self.configs[config_name]
        print(f"Setting up stack: {config.name}")
        print(f"Description: {config.description}")
        
        # Apply environment variables
        for var_name, var_value in config.environment_variables.items():
            os.environ[var_name] = var_value
            print(f"Set environment variable: {var_name}={var_value}")
        
        # For now, we'll just validate the configuration
        # In a real implementation, this would install software, configure hardware, etc.
        errors = self.validate_stack(config_name)
        if errors:
            print("Errors found during setup:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print(f"Successfully set up stack: {config.name}")
        return True


def main():
    """Example usage of the StackConfigManager."""
    manager = StackConfigManager()
    
    # List available configurations
    print("Available configurations:")
    for config_name in manager.configs.keys():
        print(f"  - {config_name}")
    
    # Set up the default configuration
    if 'research_default' in manager.configs:
        success = manager.setup_stack('research_default')
        if success:
            print("Default stack configured successfully!")
        else:
            print("Failed to configure default stack!")


if __name__ == "__main__":
    main()