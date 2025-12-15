"""
Dynamic balance controllers for the Physical AI & Humanoid Robotics course.
This module implements controllers to maintain robot balance during locomotion and manipulation.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class BalanceState:
    """Current state of the robot for balance control"""
    com_position: Tuple[float, float, float]  # Center of mass position
    com_velocity: Tuple[float, float, float]  # Center of mass velocity
    com_acceleration: Tuple[float, float, float]  # Center of mass acceleration
    angular_velocity: Tuple[float, float, float]  # Angular velocity (roll, pitch, yaw)
    angular_position: Tuple[float, float, float]  # Angular position (roll, pitch, yaw)
    support_polygon: List[Tuple[float, float]]  # Support polygon vertices (x, y)
    zmp_position: Tuple[float, float]  # Zero Moment Point position
    foot_positions: List[Tuple[float, float, float]]  # Current foot positions
    joint_angles: List[float]  # Current joint angles
    joint_velocities: List[float]  # Current joint velocities


class PIDController:
    """Generic PID controller for balance adjustments"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.prev_error = 0.0
        self.integral_error = 0.0
    
    def update(self, error: float, dt: float) -> float:
        """Update the PID controller with a new error value"""
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error
        
        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Store error for next derivative calculation
        self.prev_error = error
        
        return output


class BalanceController:
    """
    Main balance controller that integrates multiple balance strategies
    to keep the robot stable during locomotion and manipulation.
    """
    
    def __init__(self, robot_mass: float = 70.0, gravity: float = 9.81):
        self.robot_mass = robot_mass
        self.gravity = gravity
        self.com_height = 0.8  # Default CoM height for humanoid (in meters)
        
        # PID controllers for different balance aspects
        self.roll_controller = PIDController(kp=50.0, ki=0.1, kd=10.0)
        self.pitch_controller = PIDController(kp=50.0, ki=0.1, kd=10.0)
        self.lateral_controller = PIDController(kp=30.0, ki=0.05, kd=5.0)
        self.ankle_controller = PIDController(kp=20.0, ki=0.02, kd=3.0)
        
        # ZMP (Zero Moment Point) controller
        self.zmp_controller = PIDController(kp=10.0, ki=0.01, kd=2.0)
        
        # Stability thresholds
        self.roll_threshold = math.radians(10)  # 10 degrees
        self.pitch_threshold = math.radians(10)  # 10 degrees
        self.stability_margin = 0.05  # 5cm margin from support polygon edge
        
        # Balance strategy flags
        self.use_ankle_strategy = True
        self.use_hip_strategy = True
        self.use_arm_swing = True
        self.use_step_recovery = True
        
        # Current robot state
        self.current_state: Optional[BalanceState] = None
        self.is_active = False
    
    def activate(self):
        """Activate the balance controller"""
        self.is_active = True
        print("Balance controller activated")
    
    def deactivate(self):
        """Deactivate the balance controller"""
        self.is_active = False
        print("Balance controller deactivated")
    
    def update_state(self, state: BalanceState):
        """Update the current state of the robot"""
        self.current_state = state
    
    def calculate_zmp(self, com_pos: Tuple[float, float, float], 
                     com_accel: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Calculate the Zero Moment Point (ZMP) based on CoM position and acceleration
        """
        if not self.current_state:
            return (0.0, 0.0)
        
        # ZMP calculation: 
        # zmp_x = com_x - (z_height * com_acc_x) / gravity
        # zmp_y = com_y - (z_height * com_acc_y) / gravity
        zmp_x = com_pos[0] - (self.com_height * com_accel[0]) / self.gravity
        zmp_y = com_pos[1] - (self.com_height * com_accel[1]) / self.gravity
        
        return (zmp_x, zmp_y)
    
    def is_stable(self, state: BalanceState = None) -> bool:
        """
        Check if the robot is currently in a stable state
        """
        if state is None:
            state = self.current_state
        
        if state is None:
            return False
        
        # Check angular thresholds
        if abs(state.angular_position[0]) > self.roll_threshold:  # Roll
            return False
        if abs(state.angular_position[1]) > self.pitch_threshold:  # Pitch
            return False
        
        # Check ZMP position relative to support polygon
        zmp = self.calculate_zmp(state.com_position, state.com_acceleration)
        if not self._is_zmp_in_support_polygon(zmp, state.support_polygon):
            return False
        
        return True
    
    def _is_zmp_in_support_polygon(self, zmp: Tuple[float, float], 
                                  polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if the ZMP is inside the support polygon using ray casting algorithm
        """
        if len(polygon) < 3:
            return False
        
        x, z = zmp
        n = len(polygon)
        inside = False
        
        p1x, p1z = polygon[0]
        for i in range(1, n + 1):
            p2x, p2z = polygon[i % n]
            if z > min(p1z, p2z):
                if z <= max(p1z, p2z):
                    if x <= max(p1x, p2x):
                        if p1z != p2z:
                            xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1z = p2x, p2z
        
        return inside
    
    def compute_balance_correction(self, dt: float) -> Dict[str, any]:
        """
        Compute balance correction commands based on current state
        """
        if not self.is_active or not self.current_state:
            return {'active': False}
        
        state = self.current_state
        
        # Calculate current ZMP
        current_zmp = self.calculate_zmp(state.com_position, state.com_acceleration)
        
        # Find the center of support polygon as the desired ZMP
        if state.support_polygon:
            desired_zmp_x = sum(p[0] for p in state.support_polygon) / len(state.support_polygon)
            desired_zmp_y = sum(p[1] for p in state.support_polygon) / len(state.support_polygon)
            desired_zmp = (desired_zmp_x, desired_zmp_y)
        else:
            desired_zmp = (0.0, 0.0)  # Default to center if no support polygon
        
        # Compute errors
        zmp_error_x = desired_zmp[0] - current_zmp[0]
        zmp_error_y = desired_zmp[1] - current_zmp[1]
        
        # Use PID controllers to compute corrective actions
        corrective_torque_x = self.zmp_controller.update(zmp_error_x, dt)
        corrective_torque_y = self.zmp_controller.update(zmp_error_y, dt)
        
        # Compute joint adjustments for different balance strategies
        ankle_adjustments = self._compute_ankle_strategy(state, dt)
        hip_adjustments = self._compute_hip_strategy(state, dt)
        arm_adjustments = self._compute_arm_swing_strategy(state, dt)
        
        # Determine if a step is needed for recovery
        step_needed = self._is_step_needed(state)
        
        return {
            'active': True,
            'corrective_torque': (corrective_torque_x, corrective_torque_y),
            'ankle_adjustments': ankle_adjustments,
            'hip_adjustments': hip_adjustments,
            'arm_adjustments': arm_adjustments,
            'step_needed': step_needed,
            'zmp_error': (zmp_error_x, zmp_error_y)
        }
    
    def _compute_ankle_strategy(self, state: BalanceState, dt: float) -> List[float]:
        """Compute ankle joint adjustments for balance"""
        if not self.use_ankle_strategy:
            return []
        
        # Calculate roll and pitch errors
        roll_error = -state.angular_position[0]  # Negative because tilting right requires left torque
        pitch_error = -state.angular_position[1]
        
        # Use PID controllers to compute corrective torques
        roll_correction = self.roll_controller.update(roll_error, dt)
        pitch_correction = self.pitch_controller.update(pitch_error, dt)
        
        # Convert to ankle joint adjustments (simplified)
        # In a real robot, this would map to actual ankle joint commands
        ankle_adjustments = [roll_correction, pitch_correction] * 2  # For both ankles
        
        return ankle_adjustments
    
    def _compute_hip_strategy(self, state: BalanceState, dt: float) -> List[float]:
        """Compute hip joint adjustments for balance"""
        if not self.use_hip_strategy:
            return []
        
        # Calculate lateral CoM offset error
        # This is a simplified approach - in reality, hip strategy is more complex
        desired_com_x = 0.0  # Center of support polygon
        com_x_error = desired_com_x - state.com_position[0]
        
        lateral_correction = self.lateral_controller.update(com_x_error, dt)
        
        # Return hip joint adjustments (simplified)
        hip_adjustments = [lateral_correction] * 4  # 4 hip joints for a bipedal robot
        
        return hip_adjustments
    
    def _compute_arm_swing_strategy(self, state: BalanceState, dt: float) -> List[float]:
        """Compute arm adjustments to help with balance"""
        if not self.use_arm_swing:
            return []
        
        # Calculate angular momentum needed to counteract imbalance
        # Swing arms in direction opposite to tilt
        roll_compensation = -state.angular_position[0] * 0.5  # Scaled compensation
        pitch_compensation = -state.angular_position[1] * 0.5
        
        # Return arm joint adjustments (simplified)
        # In a real robot, this would involve complex inverse kinematics
        arm_adjustments = [roll_compensation, pitch_compensation] * 4  # 4 arm joints
        
        return arm_adjustments
    
    def _is_step_needed(self, state: BalanceState) -> bool:
        """Determine if a step is needed for balance recovery"""
        if not self.use_step_recovery:
            return False
        
        # A step is needed if the ZMP is outside the support polygon by more than the margin
        current_zmp = self.calculate_zmp(state.com_position, state.com_acceleration)
        in_polygon = self._is_zmp_in_support_polygon(current_zmp, state.support_polygon)
        
        if not in_polygon:
            return True
        
        # Also check if angular position exceeds safe thresholds
        if (abs(state.angular_position[0]) > self.roll_threshold * 0.8 or 
            abs(state.angular_position[1]) > self.pitch_threshold * 0.8):
            return True
        
        # Check if CoM velocity is too high in the direction of instability
        if abs(state.com_velocity[0]) > 0.5 or abs(state.com_velocity[1]) > 0.5:
            return True
        
        return False
    
    def execute_balance_control(self, dt: float) -> Dict[str, any]:
        """
        Execute the full balance control cycle
        """
        if not self.is_active or not self.current_state:
            return {'success': False, 'message': 'Balance controller not active or no state'}
        
        # Compute balance corrections
        corrections = self.compute_balance_correction(dt)
        
        # Apply corrections to robot (in a real system, this would send commands to actuators)
        # For simulation purposes, we'll just return the computed actions
        
        # Check if balance is still in danger after corrections
        is_stable = self.is_stable()
        
        result = {
            'success': True,
            'is_stable': is_stable,
            'corrections_applied': corrections,
            'message': 'Balance control executed' if is_stable else 'Balance recovery in progress'
        }
        
        # If robot is not stable and step is needed, issue warning
        if not is_stable and corrections.get('step_needed', False):
            result['message'] = 'Balance recovery required - step needed'
            result['step_required'] = True
        
        return result


class CapturePointBalancer:
    """
    Implementation of Capture Point based balance control
    The capture point is where the robot would need to step to come to a complete stop
    """
    
    def __init__(self, leg_length: float = 0.9, gravity: float = 9.81):
        self.leg_length = leg_length
        self.gravity = gravity
        self.omega = math.sqrt(gravity / leg_length)  # Natural frequency of inverted pendulum
    
    def compute_capture_point(self, com_pos: Tuple[float, float], 
                            com_vel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute the capture point based on current CoM position and velocity
        Capture Point = CoM position + CoM velocity / omega
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        
        return (cp_x, cp_y)
    
    def is_balanced(self, com_pos: Tuple[float, float], 
                   com_vel: Tuple[float, float], 
                   support_polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if the robot is balanced by comparing capture point to support polygon
        """
        capture_point = self.compute_capture_point(com_pos, com_vel)
        
        # Check if capture point is within support polygon
        return self._is_zmp_in_support_polygon(capture_point, support_polygon)
    
    def _is_zmp_in_support_polygon(self, point: Tuple[float, float], 
                                  polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside the support polygon using ray casting algorithm
        """
        if len(polygon) < 3:
            return False
        
        x, z = point
        n = len(polygon)
        inside = False
        
        p1x, p1z = polygon[0]
        for i in range(1, n + 1):
            p2x, p2z = polygon[i % n]
            if z > min(p1z, p2z):
                if z <= max(p1z, p2z):
                    if x <= max(p1x, p2x):
                        if p1z != p2z:
                            xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1z = p2x, p2z
        
        return inside