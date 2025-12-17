"""
Research-focused Isaac Sim Environment for Physical AI Experiments

This module creates advanced simulation environments specifically designed 
for conducting reproducible research experiments in the Physical AI & 
Humanoid Robotics course.
"""

import carb
import omni
import omni.kit.app
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, Sdf, UsdGeom
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import json
from pathlib import Path


class ResearchIsaacEnvironment:
    """
    Advanced Isaac Sim environment designed for reproducible research experiments.
    """
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        """
        Initialize a research-focused Isaac Sim environment.
        
        Args:
            name: Name of the environment
            config_path: Path to environment configuration file (optional)
        """
        self.name = name
        self.world = World(stage_units_in_meters=1.0)
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.objects = []
        self.tracked_metrics = {}
        self.experiment_data = []
        
        # Set up physics parameters from configuration
        self._setup_physics()
        
        # Create the environment based on configuration
        self._create_environment()
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load environment configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _default_config(self) -> Dict:
        """
        Create default configuration for research environment.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "name": self.name,
            "description": f"Research environment: {self.name}",
            "physics": {
                "gravity": -9.81,
                "solver_type": "TGS",  # TGS or PGSP
                "min_position_iteration_count": 1,
                "max_position_iteration_count": 255,
                "min_velocity_iteration_count": 1,
                "max_velocity_iteration_count": 255,
                "bounce_threshold_velocity": 0.5,
                "friction_correlation_distance": 0.001
            },
            "objects": [
                {"type": "ground_plane", "name": "ground_plane", "size": [10.0, 10.0]},
                {"type": "light", "name": "dome_light", "intensity": 3000}
            ],
            "research_scenarios": [
                {
                    "name": "object_manipulation",
                    "description": "Test manipulation of various objects with different physical properties",
                    "objects": [
                        {"name": "cube_red", "type": "cuboid", "size": [0.1, 0.1, 0.1], "position": [0.5, 0.5, 0.1], "mass": 0.5, "color": [0.8, 0.1, 0.1]},
                        {"name": "cube_blue", "type": "cuboid", "size": [0.08, 0.08, 0.08], "position": [0.7, 0.5, 0.1], "mass": 0.3, "color": [0.1, 0.1, 0.8]},
                        {"name": "sphere_green", "type": "sphere", "radius": 0.05, "position": [0.5, 0.7, 0.1], "mass": 0.2, "color": [0.1, 0.8, 0.1]}
                    ]
                }
            ]
        }
    
    def _setup_physics(self):
        """
        Configure physics parameters based on configuration.
        """
        physics_settings = self.config.get("physics", {})
        
        # Set gravity
        gravity = physics_settings.get("gravity", -9.81)
        self.world.scene.enable_gravity = True
        self.world.set_physics_dt(1.0/60.0, substeps=1)  # 60Hz physics update
        
        # Additional physics settings
        carb.log_info(f"Physics settings - Gravity: {gravity}")
    
    def _create_environment(self):
        """
        Create the environment based on configuration.
        """
        # Add ground plane
        self._add_ground_plane()
        
        # Add dome light
        self._add_dome_light()
        
        # Create research scenarios
        for scenario in self.config.get("research_scenarios", []):
            self._create_scenario(scenario)
    
    def _add_ground_plane(self):
        """
        Add a ground plane to the environment.
        """
        # Create a simple ground plane
        size = self.config["objects"][0].get("size", [10.0, 10.0])
        ground_prim = UsdGeom.Cube.Define(self.world.stage, "/World/ground_plane")
        ground_prim.GetSizeAttr().Set(100.0)  # Large plane
        
        # Transform the ground plane to be flat
        xform = UsdGeom.Xformable(ground_prim)
        xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 0.01))  # Very thin in Z direction
        xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.5))  # Position below origin
        
        # Add rigid body properties to make it static
        from omni.physx.scripts.physicsUtils import setRigidBodyProperties, setStaticCollider
    
    def _add_dome_light(self):
        """
        Add a dome light to the environment.
        """
        intensity = self.config["objects"][1].get("intensity", 3000)
        
        # Add dome light
        dome_light_path = Sdf.Path("/World/DomeLight")
        stage = self.world.stage
        dome_light = UsdGeom.DomeLight.Define(stage, dome_light_path)
        dome_light.GetIntensityAttr().Set(intensity)
    
    def _create_scenario(self, scenario: Dict):
        """
        Create a specific research scenario in the environment.
        
        Args:
            scenario: Dictionary describing the scenario to create
        """
        carb.log_info(f"Creating scenario: {scenario['name']}")
        
        for obj_config in scenario.get("objects", []):
            if obj_config["type"] == "cuboid":
                self._create_cuboid_object(obj_config)
            elif obj_config["type"] == "sphere":
                self._create_sphere_object(obj_config)
    
    def _create_cuboid_object(self, obj_config: Dict):
        """
        Create a cuboid object in the environment.
        
        Args:
            obj_config: Dictionary containing object configuration
        """
        name = obj_config["name"]
        size = obj_config["size"]
        position = obj_config["position"]
        mass = obj_config["mass"]
        color = obj_config["color"]
        
        # Create dynamic cuboid
        cuboid = self.world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=position,
                size=size[0],  # Isaac SDK uses a single size parameter for cubes
                color=np.array(color),
                mass=mass
            )
        )
        
        self.objects.append(cuboid)
        carb.log_info(f"Added cuboid: {name} at {position}")
    
    def _create_sphere_object(self, obj_config: Dict):
        """
        Create a sphere object in the environment.
        
        Args:
            obj_config: Dictionary containing object configuration
        """
        name = obj_config["name"]
        radius = obj_config["radius"]
        position = obj_config["position"]
        mass = obj_config["mass"]
        color = obj_config["color"]
        
        # For spheres in Isaac Sim, we need to create a geometry prim
        # and potentially use extensions if available
        from omni.isaac.core.utils.prims import create_primitive
        
        sphere_prim = create_primitive(
            prim_path=f"/World/{name}",
            primitive_type="Sphere",
            scale=[radius*2, radius*2, radius*2],
            position=position,
            color=color
        )
        
        self.objects.append(sphere_prim)
        carb.log_info(f"Added sphere: {name} at {position}")
    
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Reset all objects to their initial positions
        self.world.reset()
        self.experiment_data = []
        carb.log_info("Environment reset to initial state")
    
    def step(self, actions: Optional[List] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Step the environment forward in time.
        
        Args:
            actions: List of actions to apply to the environment
            
        Returns:
            Tuple of (observations, reward, done, info)
        """
        # Apply actions if provided
        if actions:
            self._apply_actions(actions)
        
        # Step the physics simulation
        self.world.step(render=True)
        
        # Collect observations
        obs = self._get_observations()
        
        # Calculate reward (for research, often just 0 or based on specific task)
        reward = 0.0
        
        # Determine if episode is done
        done = False
        
        # Additional info
        info = {"step": self.world.current_time_step_index}
        
        # Collect data for research purposes
        self._collect_step_data(obs, actions, reward, info)
        
        return obs, reward, done, info
    
    def _apply_actions(self, actions: List):
        """
        Apply actions to the environment.
        
        Args:
            actions: List of actions to apply
        """
        # In a research environment, actions might be applied to a robot
        # that is part of the scene, or to modify environment properties
        carb.log_info(f"Applying {len(actions)} actions to environment")
    
    def _get_observations(self) -> Dict:
        """
        Get observations from the environment.
        
        Returns:
            Dictionary of observations
        """
        # Get positions and states of all objects
        obs = {
            "time": self.world.current_time_step_index,
            "object_states": {}
        }
        
        for obj in self.objects:
            try:
                # Get the position of each object
                pos = obj.get_world_pose()[0]  # Get position part of pose
                obs["object_states"][obj.name] = {
                    "position": pos,
                    "orientation": obj.get_world_pose()[1],  # Get orientation part of pose
                    "linear_velocity": obj.get_linear_velocity(),
                    "angular_velocity": obj.get_angular_velocity()
                }
            except Exception as e:
                carb.log_warn(f"Could not get state for object {obj.name}: {e}")
        
        return obs
    
    def _collect_step_data(self, obs: Dict, actions: Optional[List], reward: float, info: Dict):
        """
        Collect step data for research purposes.
        
        Args:
            obs: Observations from the step
            actions: Actions applied in the step
            reward: Reward received in the step
            info: Additional info from the step
        """
        step_data = {
            "step": self.world.current_time_step_index,
            "timestamp": self.world.current_time_step_index * (1.0/60.0),  # Assuming 60Hz
            "observations": obs,
            "actions": actions if actions else [],
            "reward": reward,
            "info": info
        }
        
        self.experiment_data.append(step_data)
    
    def get_experiment_data(self) -> List[Dict]:
        """
        Get all collected experiment data.
        
        Returns:
            List of step data dictionaries
        """
        return self.experiment_data
    
    def save_experiment_data(self, filepath: str):
        """
        Save collected experiment data to a file.
        
        Args:
            filepath: Path to save the data
        """
        with open(filepath, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)
        
        carb.log_info(f"Saved experiment data to {filepath}")
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.world:
            self.world.clear()
        carb.log_info("Environment closed")


class ResearchIsaacEnvironmentBuilder:
    """
    Builder class for creating research Isaac environments with different configurations.
    """
    
    @staticmethod
    def create_object_manipulation_env() -> ResearchIsaacEnvironment:
        """
        Create an environment focused on object manipulation research.
        
        Returns:
            ResearchIsaacEnvironment configured for object manipulation
        """
        config = {
            "name": "object_manipulation_research",
            "description": "Environment for studying object manipulation in realistic physics simulation",
            "physics": {
                "gravity": -9.81,
                "solver_type": "TGS",
                "min_position_iteration_count": 4,
                "max_position_iteration_count": 8,
                "friction_correlation_distance": 0.001
            },
            "objects": [
                {"type": "ground_plane", "name": "ground_plane", "size": [5.0, 5.0]},
                {"type": "light", "name": "dome_light", "intensity": 3000}
            ],
            "research_scenarios": [
                {
                    "name": "friction_study",
                    "description": "Study effects of different friction coefficients on object manipulation",
                    "objects": [
                        {"name": "low_friction_box", "type": "cuboid", "size": [0.1, 0.1, 0.1], 
                         "position": [0.0, -0.3, 0.1], "mass": 0.5, "color": [0.8, 0.8, 0.1], 
                         "friction": 0.1},
                        {"name": "high_friction_box", "type": "cuboid", "size": [0.1, 0.1, 0.1], 
                         "position": [0.0, 0.3, 0.1], "mass": 0.5, "color": [0.8, 0.1, 0.8], 
                         "friction": 0.8}
                    ]
                },
                {
                    "name": "weight_study", 
                    "description": "Study effects of object weight on manipulation success",
                    "objects": [
                        {"name": "light_sphere", "type": "sphere", "radius": 0.05, 
                         "position": [-0.3, 0.0, 0.1], "mass": 0.1, "color": [0.1, 0.8, 0.8]},
                        {"name": "heavy_sphere", "type": "sphere", "radius": 0.05, 
                         "position": [0.3, 0.0, 0.1], "mass": 1.0, "color": [0.2, 0.6, 0.2]}
                    ]
                }
            ]
        }
        
        # Create environment with specific configuration
        env = ResearchIsaacEnvironment("object_manipulation_research")
        # Override with our specific config
        env.config = config
        env._setup_physics()
        env._create_environment()
        
        return env
    
    @staticmethod
    def create_naviation_env() -> ResearchIsaacEnvironment:
        """
        Create an environment focused on navigation research.
        
        Returns:
            ResearchIsaacEnvironment configured for navigation research
        """
        config = {
            "name": "navigation_research",
            "description": "Environment for studying robot navigation in complex scenarios",
            "physics": {
                "gravity": -9.81,
                "solver_type": "TGS",
                "min_position_iteration_count": 4,
                "max_position_iteration_count": 8
            },
            "objects": [
                {"type": "ground_plane", "name": "ground_plane", "size": [10.0, 10.0]},
                {"type": "light", "name": "dome_light", "intensity": 3000}
            ],
            "research_scenarios": [
                {
                    "name": "obstacle_avoidance",
                    "description": "Study robot navigation with dynamic and static obstacles",
                    "objects": [
                        # Static obstacles
                        {"name": "wall_1", "type": "cuboid", "size": [3.0, 0.2, 1.0], 
                         "position": [0.0, 2.0, 0.5], "mass": 10.0, "color": [0.5, 0.5, 0.5]},
                        {"name": "wall_2", "type": "cuboid", "size": [0.2, 2.0, 1.0], 
                         "position": [-1.5, 0.0, 0.5], "mass": 10.0, "color": [0.5, 0.5, 0.5]},
                        # Dynamic obstacles
                        {"name": "moving_obstacle_1", "type": "cuboid", "size": [0.3, 0.3, 0.3], 
                         "position": [1.0, 0.0, 0.2], "mass": 1.0, "color": [0.9, 0.5, 0.1]}
                    ]
                }
            ]
        }
        
        env = ResearchIsaacEnvironment("navigation_research")
        env.config = config
        env._setup_physics()
        env._create_environment()
        
        return env


def run_sample_experiment():
    """
    Run a sample research experiment using the Isaac environment.
    """
    print("Creating research environment for object manipulation...")
    
    # Create a research environment
    env = ResearchIsaacEnvironmentBuilder.create_object_manipulation_env()
    
    print("Environment created. Running sample steps...")
    
    # Run a few steps to collect data
    for i in range(100):  # 100 steps of simulation
        obs, reward, done, info = env.step()
        
        # Print progress every 20 steps
        if i % 20 == 0:
            print(f"Step {i}: Collected {len(env.get_experiment_data())} data points")
    
    print("Experiment complete.")
    
    # Save the collected data
    env.save_experiment_data("./research_experiment_data.json")
    print("Data saved to ./research_experiment_data.json")
    
    # Close the environment
    env.close()
    
    return env.get_experiment_data()


if __name__ == "__main__":
    # Run a sample experiment
    data = run_sample_experiment()
    print(f"\nCompleted sample experiment with {len(data)} steps of data collected.")