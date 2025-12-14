"""
Isaac SDK Integration Templates for the Physical AI & Humanoid Robotics Course.

This module provides templates for integrating with NVIDIA Isaac SDK, 
following the course's emphasis on GPU-accelerated perception and training
using NVIDIA Isaac for AI processing.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import os


@dataclass
class IsaacAppConfig:
    """Configuration for Isaac application."""
    app_name: str
    modules: List[str]
    assets: List[str]
    parameters: Dict[str, Any]
    build_config: str = "release"
    target_platform: str = "jetson"
    isaac_version: str = "2022.1"
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)


class IsaacModuleTemplate:
    """Template for Isaac application modules."""
    
    def __init__(self, module_name: str, namespace: str = "ai_robotics"):
        self.module_name = module_name
        self.namespace = namespace
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up a basic logger."""
        import logging
        logger = logging.getLogger(f"IsaacModuleTemplate-{self.module_name}")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_cpp_module_template(self, output_dir: str) -> Dict[str, str]:
        """
        Create C++ module templates for Isaac SDK.
        
        Args:
            output_dir: Directory to create the module files
            
        Returns:
            Dictionary mapping file names to content
        """
        module_dir = Path(output_dir) / self.module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # CMakeLists.txt
        cmake_content = f"""
cmake_minimum_required(VERSION 3.14)
project({self.module_name})

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(isaac_ros_common REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)

# Create the library
add_library({self.module_name}_nodes SHARED
  src/{self.module_name}_node.cpp
)

# Link libraries
target_link_libraries({self.module_name}_nodes
  ${{isaac_ros_common_LIBRARIES}}
)

# Install the library
install(TARGETS {self.module_name}_nodes
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

ament_export_libraries({self.module_name}_nodes)
ament_export_dependencies(rclcpp rclcpp_components std_msgs)

ament_package()
"""
        files["CMakeLists.txt"] = cmake_content
        
        # Main module node implementation
        node_content = f'''#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "isaac_ros_common/isaac_ros_utils.hpp"

namespace {self.namespace} {{

class {self.module_name.capitalize()}Node : public rclcpp::Node {{
public:
  explicit {self.module_name.capitalize()}Node(const rclcpp::NodeOptions& options);
  virtual ~{self.module_name.capitalize()}Node();

private:
  void timer_callback();
  
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
  std::string topic_name_;
}};

{self.module_name.capitalize()}Node::{self.module_name.capitalize()}Node(const rclcpp::NodeOptions& options)
  : Node("{self.module_name}_node", options), count_(0) {{
  topic_name_ = this->declare_parameter("topic_name", std::string("chatter"));
  publisher_ = this->create_publisher<std_msgs::msg::String>(topic_name_, 10);
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(500),
    std::bind(&{self.module_name.capitalize()}Node::timer_callback, this));
    
  RCLCPP_INFO(this->get_logger(), "{self.module_name.capitalize()} node initialized");
}}

{self.module_name.capitalize()}Node::~{self.module_name.capitalize()}Node() {{
  RCLCPP_INFO(this->get_logger(), "{self.module_name.capitalize()} node destroyed");
}}

void {self.module_name.capitalize()}Node::timer_callback() {{
  auto message = std_msgs::msg::String();
  message.data = "Hello from {self.module_name}! " + std::to_string(count_++);
  RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
  publisher_->publish(message);
}}

}} // namespace {self.namespace}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component
RCLCPP_COMPONENTS_REGISTER_NODE({self.namespace}::{self.module_name.capitalize()}Node)
'''
        files[f"src/{self.module_name}_node.cpp"] = node_content
        
        # Header file
        header_content = f'''#ifndef {self.module_name.upper()}_NODE_HPP
#define {self.module_name.upper()}_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace {self.namespace} {{

class {self.module_name.capitalize()}Node : public rclcpp::Node {{
public:
  explicit {self.module_name.capitalize()}Node(const rclcpp::NodeOptions& options);
  virtual ~{self.module_name.capitalize()}Node();

private:
  void timer_callback();
  
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
  std::string topic_name_;
}};

}} // namespace {self.namespace}

#endif // {self.module_name.upper()}_NODE_HPP
'''
        files[f"include/{self.module_name}_node.hpp"] = header_content
        
        # Package.xml
        package_content = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{self.module_name}</name>
  <version>1.0.0</version>
  <description>Isaac SDK integration module for {self.module_name}</description>
  <maintainer email="maintainer@nvidia.com">NVIDIA</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>std_msgs</depend>
  <depend>isaac_ros_common</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
'''
        files["package.xml"] = package_content
        
        return files


class IsaacAppTemplate:
    """Template for complete Isaac applications."""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.modules: List[IsaacModuleTemplate] = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up a basic logger."""
        import logging
        logger = logging.getLogger(f"IsaacAppTemplate-{self.app_name}")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def add_module(self, module_name: str, namespace: str = "ai_robotics") -> IsaacModuleTemplate:
        """Add a module to the application."""
        module = IsaacModuleTemplate(module_name, namespace)
        self.modules.append(module)
        return module
    
    def create_app_structure(self, output_dir: str) -> bool:
        """
        Create the complete application structure.
        
        Args:
            output_dir: Directory to create the application
            
        Returns:
            True if successful
        """
        try:
            app_dir = Path(output_dir) / self.app_name
            app_dir.mkdir(parents=True, exist_ok=True)
            
            # Create app configuration
            config = IsaacAppConfig(
                app_name=self.app_name,
                modules=[m.module_name for m in self.modules],
                assets=[],
                parameters={
                    "log_level": "INFO",
                    "use_sim_time": False
                }
            )
            
            # Write config file
            with open(app_dir / "app_config.json", "w") as f:
                f.write(config.to_json())
            
            # Create modules subdirectory and populate with each module
            modules_dir = app_dir / "modules"
            modules_dir.mkdir(exist_ok=True)
            
            for module in self.modules:
                module_dir = modules_dir / module.module_name
                module_files = module.create_cpp_module_template(str(module_dir))
                
                # Write each file
                for file_name, content in module_files.items():
                    file_path = module_dir / file_name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(content)
            
            # Create launch directory
            launch_dir = app_dir / "launch"
            launch_dir.mkdir(exist_ok=True)
            
            # Create launch file
            launch_content = f'''# Launch file for {self.app_name} Isaac application
# This file defines how to launch the application
---
launch:

- component:
    package: "{self.app_name}"
    plugin: "{self.modules[0].namespace}::{self.modules[0].module_name.capitalize()}Node"
    name: "{self.modules[0].module_name}_node"
    parameters:
      - ./config/default.yaml

# Add more components as needed
'''
            with open(launch_dir / "default.launch.yaml", "w") as f:
                f.write(launch_content)
            
            # Create config directory
            config_dir = app_dir / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Create default config
            default_config = {
                "ros__parameters": {
                    "topic_name": "chatter",
                    "rate": 10
                }
            }
            with open(config_dir / "default.yaml", "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            # Create README
            readme_content = f"""# {self.app_name} Isaac Application

This is an Isaac SDK application created for the Physical AI & Humanoid Robotics course.

## Overview
This application demonstrates integration with NVIDIA Isaac SDK for robotics applications.

## Modules
- {'; '.join([m.module_name for m in self.modules])}

## Build Instructions

```bash
cd {self.app_name}
colcon build --packages-select {self.app_name}
source install/setup.bash
```

## Run Instructions

```bash
ros2 launch {self.app_name} default.launch.yaml
```

## Configuration
See `config/default.yaml` for parameter settings.

## Directory Structure
- `modules/` - Application modules
- `launch/` - Launch files
- `config/` - Configuration files
"""
            with open(app_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            self.logger.info(f"Created Isaac application structure in {app_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating application structure: {e}")
            return False


def create_isaac_integration_example():
    """Create an example Isaac integration application."""
    # Create an application template
    app = IsaacAppTemplate("perception_pipeline")
    
    # Add modules
    perception_module = app.add_module("vision_pipeline")
    perception_module.add_module("sensor_fusion") 
    perception_module.add_module("object_detection")
    
    # Create the application structure
    success = app.create_app_structure("./isaac_apps")
    
    if success:
        print(f"Isaac application '{app.app_name}' created successfully!")
        
        # Create configuration for the application
        config = IsaacAppConfig(
            app_name="perception_pipeline",
            modules=["vision_pipeline", "sensor_fusion", "object_detection"],
            assets=[],
            parameters={
                "log_level": "INFO",
                "use_sim_time": False,
                "image_topic": "/camera/image_raw",
                "detection_threshold": 0.5
            },
            build_config="release",
            target_platform="jetson"
        )
        
        # Save the configuration
        with open("./isaac_apps/perception_pipeline/config.json", "w") as f:
            f.write(config.to_json())
        
        print("Configuration saved to ./isaac_apps/perception_pipeline/config.json")
    else:
        print("Failed to create Isaac application")


def get_isaac_sdk_info() -> Dict[str, str]:
    """Get information about Isaac SDK installation."""
    info = {
        "isaac_sdk_available": "false",
        "version": "unknown",
        "installation_path": "",
        "supported_platforms": []
    }
    
    try:
        # Check if Isaac packages are available
        result = subprocess.run([
            "dpkg", "-l", "*isaac*"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and "isaac" in result.stdout.lower():
            info["isaac_sdk_available"] = "true"
            
            # Try to get version from installed packages
            for line in result.stdout.split("\n"):
                if "isaac" in line and "ros-" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        info["version"] = parts[2]
                        break
    except Exception:
        pass
    
    # Look for Isaac installation in common locations
    common_paths = [
        "/opt/nvidia/isaac",
        "/home/jetson/isaac",
        "/usr/local/isaac"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            info["installation_path"] = path
            break
    
    info["supported_platforms"] = ["Jetson AGX Orin", "Jetson Orin NX", "Jetson Orin Nano"]
    
    return info


if __name__ == "__main__":
    # Example usage
    create_isaac_integration_example()
    
    # Print Isaac SDK information
    print("\nIsaac SDK Information:")
    print(json.dumps(get_isaac_sdk_info(), indent=2))