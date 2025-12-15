"""
Gesture recognition algorithms for the Physical AI & Humanoid Robotics course.
This module implements vision-based gesture recognition for human-robot interaction.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from enum import Enum
import math
import time
from collections import deque


class GestureType(Enum):
    """Types of recognized gestures"""
    WAVE = "wave"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STOP = "stop"
    COME_HERE = "come_here"
    FOLLOW_ME = "follow_me"
    CLAP = "clap"
    ARM_RAISE = "arm_raise"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"
    UNKNOWN = "unknown"


@dataclass
class GestureDetection:
    """Result from gesture detection"""
    gesture_type: GestureType
    confidence: float  # 0.0 to 1.0
    position: Tuple[float, float, float]  # 3D position relative to robot
    timestamp: float
    additional_info: Dict[str, Any]  # Additional gesture-specific data


@dataclass
class HandLandmark:
    """Represents a hand landmark point"""
    x: float
    y: float
    z: float  # Depth information
    visibility: float  # For pose estimation models


class GestureFeatureExtractor:
    """Extracts features from detected hand/keypoint data for gesture recognition"""
    
    def __init__(self):
        # Define key landmarks for gesture recognition
        self.landmark_indices = {
            'wrist': 0,
            'thumb_cmc': 1,
            'thumb_mcp': 2,
            'thumb_ip': 3,
            'thumb_tip': 4,
            'index_finger_mcp': 5,
            'index_finger_pip': 6,
            'index_finger_dip': 7,
            'index_finger_tip': 8,
            'middle_finger_mcp': 9,
            'middle_finger_pip': 10,
            'middle_finger_dip': 11,
            'middle_finger_tip': 12,
            'ring_finger_mcp': 13,
            'ring_finger_pip': 14,
            'ring_finger_dip': 15,
            'ring_finger_tip': 16,
            'pinky_mcp': 17,
            'pinky_pip': 18,
            'pinky_dip': 19,
            'pinky_tip': 20
        }
    
    def extract_finger_angles(self, landmarks: List[HandLandmark]) -> List[float]:
        """Extract angles between finger joints to characterize finger poses"""
        angles = []
        
        # Calculate angles for each finger
        finger_groups = [
            ['wrist', 'index_finger_pip', 'index_finger_tip'],
            ['wrist', 'middle_finger_pip', 'middle_finger_tip'],
            ['wrist', 'ring_finger_pip', 'ring_finger_tip'],
            ['wrist', 'pinky_pip', 'pinky_tip']
        ]
        
        for group in finger_groups:
            try:
                idxs = [self.landmark_indices[label] for label in group]
                if all(i < len(landmarks) for i in idxs):
                    pt1 = landmarks[idxs[0]]  # Joint 1
                    pt2 = landmarks[idxs[1]]  # Joint 2
                    pt3 = landmarks[idxs[2]]  # Joint 3
                    
                    # Calculate angle between three points
                    angle = self._calculate_angle(
                        (pt1.x, pt1.y), (pt2.x, pt2.y), (pt3.x, pt3.y)
                    )
                    angles.append(angle)
            except (IndexError, KeyError):
                angles.append(0.0)  # Default if landmarks not available
        
        return angles
    
    def _calculate_angle(self, pt1: Tuple[float, float], 
                        pt2: Tuple[float, float], 
                        pt3: Tuple[float, float]) -> float:
        """Calculate angle between three points in radians"""
        # Calculate vectors
        v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
        
        # Calculate angle between vectors using dot product
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        # Clamp to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        angle = math.acos(cos_angle)
        return angle
    
    def extract_finger_distances(self, landmarks: List[HandLandmark]) -> List[float]:
        """Extract distances between fingertips and palm to characterize hand shape"""
        distances = []
        
        # Get wrist as palm reference
        try:
            wrist_idx = self.landmark_indices['wrist']
            if wrist_idx < len(landmarks):
                wrist = landmarks[wrist_idx]
                
                # Calculate distances from wrist to each fingertip
                fingertip_labels = ['thumb_tip', 'index_finger_tip', 'middle_finger_tip', 
                                   'ring_finger_tip', 'pinky_tip']
                
                for label in fingertip_labels:
                    tip_idx = self.landmark_indices[label]
                    if tip_idx < len(landmarks):
                        tip = landmarks[tip_idx]
                        dist = math.sqrt(
                            (tip.x - wrist.x)**2 + 
                            (tip.y - wrist.y)**2 + 
                            (tip.z - wrist.z)**2
                        )
                        distances.append(dist)
        except (IndexError, KeyError):
            # Default distances if landmarks not available
            distances = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        return distances
    
    def extract_motion_features(self, landmark_sequence: List[List[HandLandmark]]) -> Dict[str, float]:
        """Extract motion-based features from a sequence of landmarks"""
        if len(landmark_sequence) < 2:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'direction': (0.0, 0.0),
                'motion_smoothness': 1.0
            }
        
        # Calculate motion of the hand's center of mass
        hand_positions = []
        for landmarks in landmark_sequence:
            if landmarks:
                # Calculate center of hand (approximate as average of all landmarks)
                x_sum = sum(lm.x for lm in landmarks)
                y_sum = sum(lm.y for lm in landmarks)
                z_sum = sum(lm.z for lm in landmarks)
                
                avg_x = x_sum / len(landmarks)
                avg_y = y_sum / len(landmarks)
                avg_z = z_sum / len(landmarks)
                
                hand_positions.append((avg_x, avg_y, avg_z))
        
        if len(hand_positions) < 2:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'direction': (0.0, 0.0),
                'motion_smoothness': 1.0
            }
        
        # Calculate average velocity
        total_displacement = 0.0
        for i in range(1, len(hand_positions)):
            dx = hand_positions[i][0] - hand_positions[i-1][0]
            dy = hand_positions[i][1] - hand_positions[i-1][1]
            dz = hand_positions[i][2] - hand_positions[i-1][2]
            displacement = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_displacement += displacement
        
        avg_velocity = total_displacement / len(hand_positions)
        
        # Calculate motion smoothness (inverse of trajectory curvature)
        smoothness = self._calculate_smoothness(hand_positions)
        
        # Calculate direction (angle of initial movement)
        dx = hand_positions[1][0] - hand_positions[0][0]
        dy = hand_positions[1][1] - hand_positions[0][1]
        direction = math.atan2(dy, dx)  # Angle in radians
        
        return {
            'velocity': avg_velocity,
            'acceleration': 0.0,  # Simplified - in practice, you'd calculate this
            'direction': direction,
            'motion_smoothness': smoothness
        }
    
    def _calculate_smoothness(self, positions: List[Tuple[float, float, float]]) -> float:
        """Calculate how smooth the motion trajectory is"""
        if len(positions) < 3:
            return 1.0  # Perfectly smooth with few points
        
        total_curvature = 0.0
        for i in range(1, len(positions) - 1):
            p1 = positions[i-1]
            p2 = positions[i]
            p3 = positions[i+1]
            
            # Calculate angle between three consecutive points
            v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
            v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
            mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            
            if mag_v1 > 0 and mag_v2 > 0:
                cos_angle = dot_product / (mag_v1 * mag_v2)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.acos(cos_angle)
                
                # Curvature is related to how far the angle is from 180 degrees (straight line)
                # 180 degrees = π radians, so curvature is π - angle
                curvature = math.pi - angle
                total_curvature += curvature
        
        # Average curvature (lower is smoother)
        avg_curvature = total_curvature / (len(positions) - 2)
        
        # Convert to smoothness (higher is smoother), normalized
        smoothness = 1.0 - min(1.0, avg_curvature / math.pi)
        return smoothness


class GestureRecognizer:
    """
    Main gesture recognition system that identifies hand gestures from keypoint data.
    Based on the requirements in the data model and API contracts.
    """
    
    def __init__(self):
        self.feature_extractor = GestureFeatureExtractor()
        self.gesture_history = deque(maxlen=10)  # Keep last 10 gesture detections
        self.confidence_threshold = 0.6
        self.motion_buffer = deque(maxlen=10)  # Buffer for motion analysis
        self.last_gesture_time = time.time()
    
    def recognize_gesture(self, landmarks: List[HandLandmark], 
                         position_3d: Tuple[float, float, float] = None) -> GestureDetection:
        """Recognize gesture from hand landmarks"""
        start_time = time.time()
        
        # Add current landmarks to motion buffer for motion analysis
        self.motion_buffer.append(landmarks)
        
        # Extract features
        finger_angles = self.feature_extractor.extract_finger_angles(landmarks)
        finger_distances = self.feature_extractor.extract_finger_distances(landmarks)
        
        # Analyze motion if we have enough frames
        motion_features = self.feature_extractor.extract_motion_features(list(self.motion_buffer))
        
        # Classify gesture based on features
        gesture_type, confidence = self._classify_gesture(
            finger_angles, finger_distances, motion_features
        )
        
        # Create detection result
        detection = GestureDetection(
            gesture_type=gesture_type,
            confidence=confidence,
            position=position_3d if position_3d else (0.0, 0.0, 0.0),
            timestamp=start_time,
            additional_info={
                'finger_angles': finger_angles,
                'finger_distances': finger_distances,
                'motion_features': motion_features
            }
        )
        
        # Add to history
        self.gesture_history.append(detection)
        
        return detection
    
    def _classify_gesture(self, finger_angles: List[float], 
                         finger_distances: List[float], 
                         motion_features: Dict[str, float]) -> Tuple[GestureType, float]:
        """Classify gesture based on extracted features"""
        # Thumbs up: thumb extended, other fingers folded
        if (finger_angles[0] > 1.5 and  # Thumb angle
            all(angle < 1.0 for angle in finger_angles[1:])):  # Other fingers folded
            return GestureType.THUMBS_UP, 0.9
        
        # Thumbs down: similar to thumbs up but potentially different orientation
        # For now, we'll use the same logic with a bit lower confidence
        if (finger_angles[0] > 1.5 and  # Thumb angle
            all(angle < 1.0 for angle in finger_angles[1:])):
            return GestureType.THUMBS_DOWN, 0.85
        
        # Stop gesture: palm facing toward robot
        # This would typically use palm orientation, simplified here
        if all(dist < 0.1 for dist in finger_distances[1:]):  # Fingers folded
            # Check if motion is minimal (stationary palm)
            if motion_features['velocity'] < 0.05:
                return GestureType.STOP, 0.85
        
        # Wave: repetitive lateral motion with fingers extended
        if motion_features['velocity'] > 0.1 and motion_features['motion_smoothness'] < 0.7:
            # Check if fingers are extended
            if all(angle > 1.0 for angle in finger_angles):
                return GestureType.WAVE, 0.8
        
        # Point: index finger extended, others folded
        if (finger_angles[1] > 1.5 and  # Index finger extended
            all(angle < 1.0 for i, angle in enumerate(finger_angles) if i != 1)):  # Others folded
            return GestureType.POINT, 0.85
        
        # Come here: palm facing robot with pulling motion
        # Simplified as waving motion toward robot
        if (motion_features['velocity'] > 0.08 and 
            abs(motion_features['direction']) > math.pi/2):  # Moving toward robot
            return GestureType.COME_HERE, 0.75
        
        # Follow me: pointing gesture with forward motion
        if (finger_angles[1] > 1.5 and  # Index finger extended
            motion_features['velocity'] > 0.05):
            return GestureType.FOLLOW_ME, 0.7
        
        # Clap: two hands coming together
        # This would require two hands detection - simplified to repetitive motion
        if (abs(motion_features['direction']) < 0.3 and  # Moving in one direction
            motion_features['motion_smoothness'] < 0.5):  # Repetitive motion
            return GestureType.CLAP, 0.7
        
        # Arm raise: all fingers extended and high position
        # Simplified as all fingers extended with high Y value
        if all(angle > 1.2 for angle in finger_angles) and len(finger_distances) > 0:
            # This would be more accurate with position data
            return GestureType.ARM_RAISE, 0.7
        
        # Default to unknown
        return GestureType.UNKNOWN, 0.0
    
    def recognize_gesture_sequence(self, landmark_sequences: List[List[HandLandmark]]) -> List[GestureDetection]:
        """Recognize gestures from a sequence of landmark frames"""
        detections = []
        
        for landmarks in landmark_sequences:
            detection = self.recognize_gesture(landmarks)
            detections.append(detection)
        
        return detections
    
    def get_recent_gestures(self, count: int = 5) -> List[GestureDetection]:
        """Get recently detected gestures"""
        return list(self.gesture_history)[-count:]


class HeadGestureRecognizer:
    """Recognizer for head-based gestures like nodding and shaking"""
    
    def __init__(self):
        self.head_positions = deque(maxlen=15)  # Track head positions over time
        self.nod_threshold = 0.1  # Threshold for detecting nods
        self.shake_threshold = 0.1  # Threshold for detecting head shakes
    
    def add_head_position(self, position: Tuple[float, float, float]):
        """Add a detected head position to the tracking buffer"""
        self.head_positions.append(position)
    
    def analyze_head_gestures(self) -> List[GestureType]:
        """Analyze head positions to detect nods and shakes"""
        if len(self.head_positions) < 5:
            return []
        
        gestures = []
        
        # Calculate movement in X (shake) and Y (nod) directions
        x_positions = [pos[0] for pos in self.head_positions]
        y_positions = [pos[1] for pos in self.head_positions]
        
        # Calculate standard deviation to detect repetitive motion
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        
        # Detect shaking (X-axis motion)
        if x_std > self.shake_threshold:
            # Check if it's a structured shake pattern
            x_range = max(x_positions) - min(x_positions)
            if x_range > 2 * self.shake_threshold:
                gestures.append(GestureType.SHAKE_HEAD)
        
        # Detect nodding (Y-axis motion)
        if y_std > self.nod_threshold:
            # Check if it's a structured nod pattern
            y_range = max(y_positions) - min(y_positions)
            if y_range > 2 * self.nod_threshold:
                gestures.append(GestureType.NOD)
        
        return gestures


class MultiModalGestureRecognizer:
    """Combines multiple gesture recognition modalities"""
    
    def __init__(self):
        self.hand_gesture_recognizer = GestureRecognizer()
        self.head_gesture_recognizer = HeadGestureRecognizer()
        self.fusion_threshold = 0.7  # Confidence threshold for accepting fused gestures
    
    def recognize_multimodal_gesture(self, hand_landmarks: List[HandLandmark],
                                   head_position: Optional[Tuple[float, float, float]] = None,
                                   face_landmarks: Optional[List] = None) -> List[GestureDetection]:
        """Recognize gestures using multiple modalities"""
        detections = []
        
        # Recognize hand gestures
        if hand_landmarks:
            hand_detection = self.hand_gesture_recognizer.recognize_gesture(hand_landmarks)
            detections.append(hand_detection)
        
        # Recognize head gestures
        if head_position:
            self.head_gesture_recognizer.add_head_position(head_position)
            head_gestures = self.head_gesture_recognizer.analyze_head_gestures()
            
            for gesture_type in head_gestures:
                detection = GestureDetection(
                    gesture_type=gesture_type,
                    confidence=0.8,  # Default confidence for head gestures
                    position=head_position,
                    timestamp=time.time(),
                    additional_info={'modality': 'head'}
                )
                detections.append(detection)
        
        return detections
    
    def get_gesture_context(self) -> Dict[str, Any]:
        """Get context information about recent gestures"""
        recent_hand_gestures = self.hand_gesture_recognizer.get_recent_gestures(count=3)
        
        return {
            'recent_hand_gestures': [
                {'type': g.gesture_type.value, 'confidence': g.confidence, 'time': g.timestamp}
                for g in recent_hand_gestures
            ],
            'gesture_sequence_active': len(recent_hand_gestures) > 0
        }


# Utility function for gesture preprocessing
def preprocess_landmarks(landmarks: List[Tuple[float, float, float]], 
                        image_width: int, image_height: int) -> List[HandLandmark]:
    """Convert raw landmark coordinates to standardized format"""
    processed_landmarks = []
    
    for x, y, z in landmarks:
        # Normalize coordinates relative to image dimensions
        norm_x = x / image_width if image_width > 0 else x
        norm_y = y / image_height if image_height > 0 else y
        
        processed_landmarks.append(HandLandmark(
            x=norm_x,
            y=norm_y,
            z=z,
            visibility=1.0  # Assume full visibility for now
        ))
    
    return processed_landmarks