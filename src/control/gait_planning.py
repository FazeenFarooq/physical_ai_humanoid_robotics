"""
Gait planning algorithms for the Physical AI & Humanoid Robotics course.
This module implements algorithms for planning and controlling robot gaits.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class GaitPhase:
    """Represents a phase in a gait cycle"""
    name: str
    duration: float  # Duration in seconds
    support_feet: List[str]  # Which feet are in contact with ground
    swing_feet: List[str]  # Which feet are swinging
    target_velocities: Dict[str, Tuple[float, float, float]]  # Target velocities for each foot in this phase


@dataclass
class GaitPattern:
    """Defines a complete gait pattern"""
    name: str
    phases: List[GaitPhase]
    cycle_duration: float  # Total duration of one gait cycle
    base_velocity: float  # Base velocity for this gait
    foot_positions: Dict[str, Tuple[float, float, float]]  # Nominal foot positions in stance phase
    stability_margin: float  # Minimum stability margin for this gait


class GaitPlanner:
    """
    Planner for robot gaits including pattern selection, 
    trajectory generation, and stability checking.
    """
    
    def __init__(self, robot_mass: float = 70.0):  # Default humanoid mass 70kg
        self.robot_mass = robot_mass
        self.gravity = 9.81  # m/s^2
        self.gait_patterns = self._initialize_gait_patterns()
        self.current_gait: Optional[GaitPattern] = None
        self.current_phase_idx = 0
        self.phase_time = 0.0
    
    def _initialize_gait_patterns(self) -> Dict[str, GaitPattern]:
        """Initialize the available gait patterns for the robot"""
        patterns = {}
        
        # Walk gait pattern
        walk_phases = [
            GaitPhase(
                name="double_support_1",
                duration=0.1,
                support_feet=["left", "right"],
                swing_feet=[],
                target_velocities={}
            ),
            GaitPhase(
                name="left_swing",
                duration=0.4,
                support_feet=["right"],
                swing_feet=["left"],
                target_velocities={"left": (0.0, 0.0, 0.0)}
            ),
            GaitPhase(
                name="double_support_2",
                duration=0.1,
                support_feet=["left", "right"],
                swing_feet=[],
                target_velocities={}
            ),
            GaitPhase(
                name="right_swing",
                duration=0.4,
                support_feet=["left"],
                swing_feet=["right"],
                target_velocities={"right": (0.0, 0.0, 0.0)}
            )
        ]
        
        walk_gait = GaitPattern(
            name="walk",
            phases=walk_phases,
            cycle_duration=1.0,
            base_velocity=0.5,  # m/s
            foot_positions={
                "left": (0.0, 0.1, 0.0),   # Nominal position relative to body
                "right": (0.0, -0.1, 0.0)
            },
            stability_margin=0.05  # 5cm margin
        )
        
        patterns["walk"] = walk_gait
        
        # Trot gait pattern
        trot_phases = [
            GaitPhase(
                name="left_front_right_back_swing",
                duration=0.25,
                support_feet=["right_front", "left_back"],
                swing_feet=["left_front", "right_back"],
                target_velocities={"left_front": (0.0, 0.0, 0.0), "right_back": (0.0, 0.0, 0.0)}
            ),
            GaitPhase(
                name="double_support",
                duration=0.1,
                support_feet=["left_front", "right_front", "left_back", "right_back"],
                swing_feet=[],
                target_velocities={}
            ),
            GaitPhase(
                name="right_front_left_back_swing",
                duration=0.25,
                support_feet=["left_front", "right_back"],
                swing_feet=["right_front", "left_back"],
                target_velocities={"right_front": (0.0, 0.0, 0.0), "left_back": (0.0, 0.0, 0.0)}
            )
        ]
        
        trot_gait = GaitPattern(
            name="trot",
            phases=trot_phases,
            cycle_duration=0.6,
            base_velocity=1.0,  # m/s
            foot_positions={
                "left_front": (0.2, 0.1, 0.0),
                "right_front": (0.2, -0.1, 0.0),
                "left_back": (-0.2, 0.1, 0.0),
                "right_back": (-0.2, -0.1, 0.0)
            },
            stability_margin=0.08
        )
        
        patterns["trot"] = trot_gait
        
        # Crawl gait pattern
        crawl_phases = [
            GaitPhase(
                name="left_front_swing",
                duration=0.3,
                support_feet=["right_front", "left_back", "right_back"],
                swing_feet=["left_front"],
                target_velocities={"left_front": (0.0, 0.0, 0.0)}
            ),
            GaitPhase(
                name="left_back_swing",
                duration=0.3,
                support_feet=["right_front", "left_front", "right_back"],
                swing_feet=["left_back"],
                target_velocities={"left_back": (0.0, 0.0, 0.0)}
            ),
            GaitPhase(
                name="right_front_swing",
                duration=0.3,
                support_feet=["left_front", "left_back", "right_back"],
                swing_feet=["right_front"],
                target_velocities={"right_front": (0.0, 0.0, 0.0)}
            ),
            GaitPhase(
                name="right_back_swing",
                duration=0.3,
                support_feet=["left_front", "left_back", "right_front"],
                swing_feet=["right_back"],
                target_velocities={"right_back": (0.0, 0.0, 0.0)}
            )
        ]
        
        crawl_gait = GaitPattern(
            name="crawl",
            phases=crawl_phases,
            cycle_duration=1.2,
            base_velocity=0.3,  # m/s
            foot_positions={
                "left_front": (0.2, 0.1, 0.0),
                "right_front": (0.2, -0.1, 0.0),
                "left_back": (-0.2, 0.1, 0.0),
                "right_back": (-0.2, -0.1, 0.0)
            },
            stability_margin=0.1  # Larger margin for stability
        )
        
        patterns["crawl"] = crawl_gait
        
        return patterns
    
    def select_gait(self, desired_velocity: float, terrain_type: str = "flat") -> Optional[str]:
        """Select an appropriate gait based on desired velocity and terrain"""
        if terrain_type == "rough":
            # For rough terrain, use crawl gait for maximum stability
            return "crawl"
        elif terrain_type == "slippery":
            # For slippery terrain, use walk gait for better control
            return "walk"
        else:
            # For normal terrain, select gait based on desired speed
            if desired_velocity < 0.6:
                return "walk"
            elif desired_velocity < 1.2:
                return "trot"
            else:
                # For higher speeds, would need gallop or run patterns
                # For now, return the fastest available gait
                return "trot"
    
    def get_current_gait_pattern(self) -> Optional[GaitPattern]:
        """Get the current active gait pattern"""
        return self.current_gait
    
    def get_current_phase(self) -> Optional[GaitPhase]:
        """Get the current active gait phase"""
        if self.current_gait is None or self.current_phase_idx >= len(self.current_gait.phases):
            return None
        return self.current_gait.phases[self.current_phase_idx]
    
    def plan_foot_trajectory(self, foot_name: str, start_pos: Tuple[float, float, float], 
                           end_pos: Tuple[float, float, float], 
                           phase_duration: float, 
                           trajectory_type: str = "cubic") -> List[Tuple[float, float, float]]:
        """Plan a smooth trajectory for a foot between start and end positions"""
        if trajectory_type == "cubic":
            # Create a cubic trajectory with lift-off and landing
            trajectory = []
            steps = int(phase_duration * 100)  # 100 steps per second
            
            # Calculate lift height - 5cm above ground is sufficient for walking
            lift_height = 0.05
            
            start_x, start_y, start_z = start_pos
            end_x, end_y, end_z = end_pos
            
            for i in range(steps + 1):
                t = i / steps  # Parameter from 0 to 1
                
                # Cubic interpolation for smooth movement
                x = start_x + (end_x - start_x) * t
                y = start_y + (end_y - start_y) * t
                
                # Parabolic lift trajectory - rise and fall during swing phase
                z = start_z
                if t < 0.5:
                    # Rising phase
                    z_lift = start_z + lift_height * (4 * t * t)  # Quadratic rise
                else:
                    # Falling phase
                    z_lift = start_z + lift_height * (4 * (1 - t) * (1 - t))  # Quadratic fall
                
                # Use the lifted Z value only if foot is in swing phase
                # For now, simulate that this foot is in swing phase
                trajectory.append((x, y, z_lift))
            
            return trajectory
        else:
            # Linear trajectory as fallback
            trajectory = []
            steps = int(phase_duration * 100)
            start_x, start_y, start_z = start_pos
            end_x, end_y, end_z = end_pos
            
            for i in range(steps + 1):
                t = i / steps
                x = start_x + (end_x - start_x) * t
                y = start_y + (end_y - start_y) * t
                z = start_z + (end_z - start_z) * t
                trajectory.append((x, y, z))
            
            return trajectory
    
    def calculate_stability_margin(self, foot_positions: Dict[str, Tuple[float, float, float]]) -> float:
        """Calculate the stability margin based on foot positions and CoM"""
        if not foot_positions:
            return 0.0
        
        # Calculate center of support polygon (convex hull of feet in contact)
        # For simplicity, we'll calculate a 2D support polygon (x, y)
        x_coords = [pos[0] for pos in foot_positions.values()]
        y_coords = [pos[1] for pos in foot_positions.values()]
        
        # Calculate centroid of support polygon
        if x_coords and y_coords:
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Calculate minimum distance from center to any support foot
            min_distance = float('inf')
            for x, y, _ in foot_positions.values():
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                min_distance = min(min_distance, dist)
            
            # This is a simplified calculation
            # In reality, this would involve computing the actual support polygon
            # and checking if the center of mass projection is inside it
            return min_distance
        else:
            return 0.0
    
    def is_gait_stable(self, gait_name: str, com_pos: Tuple[float, float, float]) -> bool:
        """Check if a given gait is stable with the robot's center of mass position"""
        if gait_name not in self.gait_patterns:
            return False
        
        gait = self.gait_patterns[gait_name]
        
        # Check if the CoM is within the stability margin of the nominal foot positions
        # This is a simplified check
        for foot_name, foot_pos in gait.foot_positions.items():
            # Calculate distance from CoM to nominal foot position
            dist = math.sqrt(
                (com_pos[0] - foot_pos[0])**2 + 
                (com_pos[1] - foot_pos[1])**2
            )
            
            # If CoM is too far from all foot positions, gait may not be stable
            if dist <= gait.stability_margin:
                return True
        
        return False
    
    def update_gait_phase(self, dt: float) -> bool:
        """Update the current gait phase based on elapsed time"""
        if self.current_gait is None:
            return False
        
        # Update the phase time
        self.phase_time += dt
        
        # Check if we need to move to the next phase
        current_phase = self.current_gait.phases[self.current_phase_idx]
        if self.phase_time >= current_phase.duration:
            # Move to next phase
            self.phase_time = 0.0
            self.current_phase_idx = (self.current_phase_idx + 1) % len(self.current_gait.phases)
            return True  # Phase changed
        
        return False  # Phase did not change
    
    def set_desired_velocity(self, desired_velocity: float, terrain_type: str = "flat"):
        """Set the desired velocity and select appropriate gait"""
        gait_name = self.select_gait(desired_velocity, terrain_type)
        if gait_name and gait_name in self.gait_patterns:
            self.current_gait = self.gait_patterns[gait_name]
            self.current_phase_idx = 0
            self.phase_time = 0.0