"""
Kinematic models for humanoid robots in the Physical AI & Humanoid Robotics course.
This module implements forward and inverse kinematics for humanoid robots.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class Joint:
    """Represents a joint in the robot kinematic chain"""
    name: str
    type: str  # 'revolute', 'prismatic', 'fixed'
    position: Tuple[float, float, float]  # Position in 3D space
    orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    limits: Tuple[float, float]  # Min and max joint angle/range
    parent: str = None  # Parent joint name
    children: List[str] = None  # Child joint names


@dataclass
class Link:
    """Represents a rigid link between joints in the kinematic chain"""
    name: str
    length: float
    mass: float
    com: Tuple[float, float, float]  # Center of mass position relative to joint
    inertia: Tuple[float, float, float]  # Moments of inertia (Ixx, Iyy, Izz)


class KinematicChain:
    """A kinematic chain representing part of a robot (e.g., arm, leg)"""
    
    def __init__(self, name: str, joints: List[Joint], links: List[Link]):
        self.name = name
        self.joints = joints
        self.links = links
        self.joint_angles = [0.0] * len(joints)  # Current joint angles
    
    def set_joint_angles(self, angles: List[float]):
        """Set the joint angles for the kinematic chain"""
        if len(angles) != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} angles, got {len(angles)}")
        
        for i, angle in enumerate(angles):
            joint = self.joints[i]
            # Check limits
            if angle < joint.limits[0] or angle > joint.limits[1]:
                raise ValueError(f"Angle {angle} for joint {joint.name} is outside limits {joint.limits}")
        
        self.joint_angles = angles[:]
    
    def forward_kinematics(self, joint_angles: List[float] = None) -> List[Tuple[float, float, float]]:
        """Calculate the end positions of each link in the chain"""
        if joint_angles is None:
            joint_angles = self.joint_angles
        
        positions = []
        current_pos = [0.0, 0.0, 0.0]  # Starting from base
        
        for i, joint in enumerate(self.joints):
            # For this simplified implementation, we'll calculate based on joint angles
            # A real implementation would use transformation matrices
            
            # Calculate position based on joint angle and link length
            angle = joint_angles[i] if i < len(joint_angles) else 0.0
            
            if i == 0:
                # First joint - use its position
                current_pos = list(joint.position)
            else:
                # Calculate relative position based on previous joint and current angle
                # This is a simplified 2D representation in XZ plane
                if i < len(self.links):
                    link_length = self.links[i-1].length
                    current_pos[0] += link_length * math.cos(angle)
                    current_pos[2] += link_length * math.sin(angle)
            
            positions.append(tuple(current_pos))
        
        return positions
    
    def inverse_kinematics(self, target_pos: Tuple[float, float, float], 
                          max_iterations: int = 100, tolerance: float = 0.001) -> List[float]:
        """Calculate joint angles to reach a target position using Jacobian transpose method"""
        # Initial joint angles
        current_angles = self.joint_angles[:]
        
        for iteration in range(max_iterations):
            # Calculate current end effector position
            current_pos = self.forward_kinematics(current_angles)[-1] if self.forward_kinematics(current_angles) else (0, 0, 0)
            
            # Calculate error
            error = [
                target_pos[0] - current_pos[0],
                target_pos[1] - current_pos[1],
                target_pos[2] - current_pos[2]
            ]
            
            # Check if we're close enough
            error_magnitude = math.sqrt(sum(e**2 for e in error))
            if error_magnitude < tolerance:
                return current_angles
            
            # Calculate Jacobian (simplified)
            jacobian = self._calculate_jacobian(current_angles)
            
            # Use Jacobian transpose method to compute angle adjustments
            # This is a simplified implementation
            angle_deltas = [0.0] * len(self.joints)
            for i in range(len(self.joints)):
                if i < len(jacobian):
                    # Simplified Jacobian transpose calculation
                    for j in range(min(3, len(jacobian[i]))):  # x, y, z components
                        angle_deltas[i] += jacobian[i][j] * error[j]
            
            # Update angles
            for i in range(len(current_angles)):
                new_angle = current_angles[i] + angle_deltas[i] * 0.1  # Learning rate
                # Apply joint limits
                new_angle = max(self.joints[i].limits[0], min(self.joints[i].limits[1], new_angle))
                current_angles[i] = new_angle
        
        # Return best solution found
        return current_angles
    
    def _calculate_jacobian(self, joint_angles: List[float]) -> List[List[float]]:
        """Calculate the Jacobian matrix for the kinematic chain"""
        # This is a simplified implementation
        # For a real robot, this would require more complex transformation calculations
        num_joints = len(self.joints)
        jacobian = [[0.0 for _ in range(3)] for _ in range(num_joints)]  # 3 for x, y, z
        
        # Calculate partial derivatives for each joint
        # This is highly simplified for the example
        for i in range(num_joints):
            # Simplified calculation - in reality, each column would represent 
            # the partial derivative of end-effector position w.r.t each joint angle
            if i < len(self.links):
                link_length = self.links[i].length
                angle = joint_angles[i] if i < len(joint_angles) else 0.0
                
                # Partial derivatives
                jacobian[i][0] = -link_length * math.sin(angle)  # dx/dtheta
                jacobian[i][2] = link_length * math.cos(angle)   # dz/dtheta
        
        return jacobian


class HumanoidKinematicModel:
    """Complete kinematic model for a humanoid robot"""
    
    def __init__(self):
        self.chains: Dict[str, KinematicChain] = {}
        self.base_position = (0.0, 0.0, 0.0)
        self.base_orientation = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    
    def add_chain(self, chain: KinematicChain):
        """Add a kinematic chain to the humanoid model"""
        self.chains[chain.name] = chain
    
    def get_chain(self, name: str) -> KinematicChain:
        """Get a specific kinematic chain by name"""
        return self.chains.get(name)
    
    def set_base_pose(self, position: Tuple[float, float, float], 
                     orientation: Tuple[float, float, float, float]):
        """Set the base position and orientation of the robot"""
        self.base_position = position
        self.base_orientation = orientation
    
    def calculate_center_of_mass(self) -> Tuple[float, float, float]:
        """Calculate the center of mass of the robot"""
        total_mass = 0.0
        weighted_pos = [0.0, 0.0, 0.0]
        
        # Calculate CoM based on link masses and positions
        for chain in self.chains.values():
            for i, link in enumerate(chain.links):
                pos = chain.forward_kinematics()[i] if i < len(chain.forward_kinematics()) else (0, 0, 0)
                
                total_mass += link.mass
                weighted_pos[0] += pos[0] * link.mass
                weighted_pos[1] += pos[1] * link.mass
                weighted_pos[2] += pos[2] * link.mass
        
        if total_mass > 0:
            com = (
                weighted_pos[0] / total_mass,
                weighted_pos[1] / total_mass,
                weighted_pos[2] / total_mass
            )
        else:
            com = (0.0, 0.0, 0.0)
        
        return com
    
    def is_stable(self) -> bool:
        """Check if the robot's center of mass is within its support polygon"""
        # Calculate CoM
        com = self.calculate_center_of_mass()
        
        # Simplified stability check - in reality, this would involve 
        # calculating the support polygon from ground contact points
        # and checking if CoM projection is inside it
        return abs(com[0]) < 0.2 and abs(com[1]) < 0.2  # Simplified check
    
    def get_end_effector_positions(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Get end effector positions for all kinematic chains"""
        positions = {}
        for name, chain in self.chains.items():
            positions[name] = chain.forward_kinematics()
        return positions