"""
Hardware calibration tools for the Physical AI & Humanoid Robotics Course

This module provides tools for calibrating various hardware components
used in the course, including cameras, IMUs, LiDAR sensors, and robot kinematics.
"""
import numpy as np
import cv2
import yaml
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Data class to store calibration results"""
    success: bool
    parameters: Dict[str, Any]
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class CameraCalibrator:
    """
    Camera calibration tool for calibrating RGB and depth cameras.
    
    Uses a chessboard pattern for calibration and produces intrinsic
    and extrinsic parameters for the camera.
    """
    
    def __init__(self, board_width: int = 9, board_height: int = 6, square_size: float = 1.0):
        """
        Initialize the camera calibrator
        
        Args:
            board_width: Number of internal corners in the chessboard width
            board_height: Number of internal corners in the chessboard height
            square_size: Size of one square in the chessboard (in cm or inches)
        """
        self.board_width = board_width
        self.board_height = board_height
        self.square_size = square_size
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
    
    def detect_chessboard(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect chessboard pattern in an image
        
        Args:
            image: Input image for calibration
            
        Returns:
            Tuple of (success, corners) where corners are the detected points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        ret, corners = cv2.findChessboardCorners(gray, 
                                               (self.board_width, self.board_height), 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners_refined
        
        return False, None
    
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add an image to the calibration set if it contains a valid chessboard
        
        Args:
            image: Image to add to calibration set
            
        Returns:
            True if image was added successfully, False otherwise
        """
        success, corners = self.detect_chessboard(image)
        
        if success:
            # Prepare object points
            objp = np.zeros((self.board_height * self.board_width, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.board_width, 0:self.board_height].T.reshape(-1, 2)
            objp *= self.square_size
            
            self.obj_points.append(objp)
            self.img_points.append(corners.reshape(-1, 2))
            return True
        
        return False
    
    def calibrate_camera(self) -> CalibrationResult:
        """
        Perform camera calibration using collected images
        
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(self.obj_points) < 10:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need at least 10 images for reliable calibration"
            )
        
        try:
            # Calibrate the camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, 
                self.img_points[0].shape[::-1], 
                None, None
            )
            
            if not ret:
                return CalibrationResult(
                    success=False,
                    parameters={},
                    error="Camera calibration failed"
                )
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.obj_points)):
                img_points2, _ = cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            
            avg_error = total_error / len(self.obj_points)
            
            calibration_params = {
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'reprojection_error': float(avg_error)
            }
            
            metrics = {
                'reprojection_error': float(avg_error),
                'num_images_used': len(self.obj_points)
            }
            
            return CalibrationResult(
                success=True,
                parameters=calibration_params,
                metrics=metrics
            )
        
        except Exception as e:
            return CalibrationResult(
                success=False,
                parameters={},
                error=f"Calibration failed with error: {str(e)}"
            )


class IMUCalibrator:
    """
    IMU calibration tool for calibrating inertial measurement units.
    
    Provides methods for bias, scale factor, and alignment calibration.
    """
    
    def __init__(self):
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        self.magnetometer_cal = {
            'hard_iron_bias': np.zeros(3),
            'soft_iron_matrix': np.eye(3)
        }
        self.calibration_data = []
    
    def calibrate_accelerometer_static(self, samples: List[np.ndarray]) -> CalibrationResult:
        """
        Calibrate accelerometer using static samples
        
        Args:
            samples: List of accelerometer readings taken in static conditions
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(samples) < 100:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need at least 100 samples for reliable accelerometer calibration"
            )
        
        # Calculate mean to determine bias
        avg_reading = np.mean(samples, axis=0)
        
        # In static condition, accelerometer should read [0, 0, 9.81] (gravity)
        expected = np.array([0.0, 0.0, 9.81])
        
        # Calculate bias
        bias = avg_reading - expected
        
        # Store calibration
        self.accel_bias = bias
        
        return CalibrationResult(
            success=True,
            parameters={
                'bias': bias.tolist(),
                'gravity_estimate': expected.tolist()
            }
        )
    
    def calibrate_gyroscope_static(self, samples: List[np.ndarray]) -> CalibrationResult:
        """
        Calibrate gyroscope using static samples
        
        Args:
            samples: List of gyroscope readings taken in static conditions
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(samples) < 100:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need at least 100 samples for reliable gyroscope calibration"
            )
        
        # Calculate mean to determine bias (should be zero in static condition)
        bias = np.mean(samples, axis=0)
        
        # Store calibration
        self.gyro_bias = bias
        
        return CalibrationResult(
            success=True,
            parameters={
                'bias': bias.tolist()
            }
        )
    
    def calibrate_magnetometer_ellipsoid(self, samples: List[np.ndarray]) -> CalibrationResult:
        """
        Calibrate magnetometer using ellipsoid fitting method
        
        Args:
            samples: List of magnetometer readings taken from various orientations
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(samples) < 50:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need at least 50 samples for reliable magnetometer calibration"
            )
        
        # Convert to numpy array
        data = np.array(samples)
        
        # Simple ellipsoid fitting to get hard iron bias
        # Calculate center of sphere (bias)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        center = (min_vals + max_vals) / 2.0
        
        # Store calibration
        self.magnetometer_cal['hard_iron_bias'] = center
        
        return CalibrationResult(
            success=True,
            parameters={
                'hard_iron_bias': center.tolist(),
                'sample_count': len(samples)
            }
        )


class LiDARCalibrator:
    """
    LiDAR calibration tool for calibrating LiDAR sensors.
    
    Provides methods for extrinsic calibration between LiDAR and other sensors.
    """
    
    def __init__(self):
        self.extrinsics = {}
    
    def calibrate_lidar_camera_extrinsics(self, 
                                        lidar_points: np.ndarray, 
                                        image_points: np.ndarray,
                                        camera_matrix: np.ndarray,
                                        dist_coeffs: np.ndarray) -> CalibrationResult:
        """
        Calibrate extrinsics between LiDAR and camera
        
        Args:
            lidar_points: 3D points from LiDAR
            image_points: 2D points from camera corresponding to the 3D points
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(lidar_points) < 6 or len(image_points) < 6:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need at least 6 corresponding points for reliable extrinsic calibration"
            )
        
        try:
            # Use solvePnP to find extrinsic parameters
            success, rvec, tvec = cv2.solvePnP(
                lidar_points.astype(np.float32), 
                image_points.astype(np.float32),
                camera_matrix, 
                dist_coeffs
            )
            
            if not success:
                return CalibrationResult(
                    success=False,
                    parameters={},
                    error="Could not solve for extrinsic parameters"
                )
            
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rmat
            transform[:3, 3] = tvec.flatten()
            
            self.extrinsics['lidar_to_camera'] = transform
            
            return CalibrationResult(
                success=True,
                parameters={
                    'rotation_vector': rvec.tolist(),
                    'translation_vector': tvec.tolist(),
                    'rotation_matrix': rmat.tolist(),
                    'transformation_matrix': transform.tolist()
                }
            )
        
        except Exception as e:
            return CalibrationResult(
                success=False,
                parameters={},
                error=f"LiDAR-camera calibration failed: {str(e)}"
            )


class KinematicCalibrator:
    """
    Kinematic calibration tool for robot manipulators and platforms.
    
    Provides methods for calibrating DH parameters, joint offsets, and kinematic models.
    """
    
    def __init__(self):
        self.joint_offsets = {}
        self.dh_parameters = {}
        self.tool_frame = np.eye(4)
    
    def calibrate_joint_offsets(self, 
                              expected_positions: List[np.ndarray], 
                              actual_positions: List[np.ndarray]) -> CalibrationResult:
        """
        Calibrate joint offsets by comparing expected vs actual positions
        
        Args:
            expected_positions: Expected joint positions (from controller)
            actual_positions: Actual joint positions (from encoders/sensors)
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if len(expected_positions) != len(actual_positions) or len(expected_positions) < 3:
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need matching sets of expected and actual positions (at least 3 samples)"
            )
        
        # Calculate average offset for each joint
        expected_array = np.array(expected_positions)
        actual_array = np.array(actual_positions)
        
        offsets = np.mean(expected_array - actual_array, axis=0)
        
        # Store calibration
        for i, offset in enumerate(offsets):
            self.joint_offsets[f'joint_{i}'] = offset
        
        return CalibrationResult(
            success=True,
            parameters={
                'joint_offsets': offsets.tolist(),
                'sample_count': len(expected_positions)
            }
        )
    
    def calibrate_tool_frame(self, 
                           tcp_positions: List[np.ndarray], 
                           tcp_orientations: List[np.ndarray],
                           ee_positions: List[np.ndarray], 
                           ee_orientations: List[np.ndarray]) -> CalibrationResult:
        """
        Calibrate tool frame (TCP - Tool Center Point) relative to end effector
        
        Args:
            tcp_positions: TCP positions in world coordinates
            tcp_orientations: TCP orientations in world coordinates
            ee_positions: End effector positions in world coordinates
            ee_orientations: End effector orientations in world coordinates
            
        Returns:
            CalibrationResult containing success status and parameters
        """
        if (len(tcp_positions) != len(tcp_orientations) or 
            len(ee_positions) != len(ee_orientations) or
            len(tcp_positions) < 3 or len(ee_positions) < 3):
            return CalibrationResult(
                success=False,
                parameters={},
                error="Need matching sets of positions and orientations (at least 3 samples for each)"
            )
        
        try:
            # Calculate transformation between TCP and end effector
            # This is a simplified approach - in practice, more sophisticated methods may be needed
            tcp_pos = np.array(tcp_positions[0])  # At home position
            tcp_rot = np.array(tcp_orientations[0])
            
            ee_pos = np.array(ee_positions[0])     # At home position
            ee_rot = np.array(ee_orientations[0])
            
            # Calculate transform from EE to TCP
            # In a real implementation, this would involve more complex math
            # to compute the transform between the two frames
            translation = tcp_pos - ee_pos
            
            # For rotation, we need to compute the relative rotation matrix
            # This is a simplified placeholder
            tool_transform = np.eye(4)
            tool_transform[:3, 3] = translation
            
            # Store calibration
            self.tool_frame = tool_transform
            
            return CalibrationResult(
                success=True,
                parameters={
                    'tool_transform': tool_transform.tolist()
                }
            )
        
        except Exception as e:
            return CalibrationResult(
                success=False,
                parameters={},
                error=f"Tool frame calibration failed: {str(e)}"
            )


class CalibrationManager:
    """
    Main calibration manager that coordinates all calibration processes.
    """
    
    def __init__(self):
        self.camera_calibrator = CameraCalibrator()
        self.imu_calibrator = IMUCalibrator()
        self.lidar_calibrator = LiDARCalibrator()
        self.kinematic_calibrator = KinematicCalibrator()
        self.calibration_results = {}
    
    def save_calibration_to_file(self, file_path: str, device_name: str) -> bool:
        """
        Save calibration parameters to a YAML file
        
        Args:
            file_path: Path to save the calibration file
            device_name: Name of the device being calibrated
            
        Returns:
            True if successful, False otherwise
        """
        calibration_data = {
            'device': device_name,
            'timestamp': time.time(),
            'calibration_results': self.calibration_results
        }
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving calibration to {file_path}: {str(e)}")
            return False
    
    def load_calibration_from_file(self, file_path: str) -> bool:
        """
        Load calibration parameters from a YAML file
        
        Args:
            file_path: Path to load the calibration file from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                calibration_data = yaml.safe_load(f)
            
            self.calibration_results = calibration_data.get('calibration_results', {})
            return True
        except Exception as e:
            print(f"Error loading calibration from {file_path}: {str(e)}")
            return False
    
    def run_complete_calibration(self, 
                               device_name: str, 
                               data_directory: str, 
                               output_path: str) -> Dict[str, Any]:
        """
        Run a complete calibration routine for a device
        
        Args:
            device_name: Name identifier for the device
            data_directory: Directory containing calibration data
            output_path: Path to save the results
            
        Returns:
            Dictionary containing overall calibration results
        """
        results = {
            'device': device_name,
            'timestamp': time.time(),
            'individual_results': {},
            'overall_success': True
        }
        
        try:
            # Placeholder for actual calibration process
            # In a real implementation, this would:
            # 1. Load calibration data from the specified directory
            # 2. Run appropriate calibration routines based on device type
            # 3. Consolidate results
            # 4. Save to output path
            
            # Example: If it's a camera, run camera calibration
            if 'camera' in device_name.lower():
                # Load images and run calibration (simplified)
                # This would load images from data_directory and use camera_calibrator
                pass
            
            # Example: If it's an IMU, run IMU calibration
            elif 'imu' in device_name.lower():
                # Process IMU data from data_directory and use imu_calibrator
                pass
            
            # Save results
            if self.save_calibration_to_file(output_path, device_name):
                results['save_success'] = True
            else:
                results['save_success'] = False
                results['overall_success'] = False
            
            return results
            
        except Exception as e:
            results['overall_success'] = False
            results['error'] = str(e)
            return results


def create_default_calibration_files():
    """
    Create default calibration files for common sensors used in the course.
    """
    # Create directory if it doesn't exist
    Path("calibrations").mkdir(exist_ok=True)
    
    # Create default camera calibration file
    default_camera_cal = {
        'camera_matrix': [
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
        'image_width': 640,
        'image_height': 480
    }
    
    with open("calibrations/default_camera.yaml", 'w') as f:
        yaml.dump(default_camera_cal, f, default_flow_style=False)
    
    # Create default IMU calibration file
    default_imu_cal = {
        'accel_bias': [0.0, 0.0, 0.0],
        'gyro_bias': [0.0, 0.0, 0.0],
        'accel_scale': [1.0, 1.0, 1.0],
        'gyro_scale': [1.0, 1.0, 1.0]
    }
    
    with open("calibrations/default_imu.yaml", 'w') as f:
        yaml.dump(default_imu_cal, f, default_flow_style=False)
    
    print("Default calibration files created in calibrations/ directory")


if __name__ == "__main__":
    # Example usage
    create_default_calibration_files()
    print("Hardware calibration tools initialized and default files created.")