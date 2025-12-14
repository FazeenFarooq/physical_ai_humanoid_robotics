"""
Multi-modal Perception Pipeline for the Physical AI & Humanoid Robotics Course.

This module implements a pipeline that processes multiple sensor modalities 
(RGB, depth, LiDAR, audio, IMU) to create a unified perception of the environment
for vision-language-action (VLA) systems.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import threading
import queue
import time
import logging
from pathlib import Path

# Import for multimodal processing
from PIL import Image
import torchvision.transforms as transforms


@dataclass
class PerceptionInput:
    """Input data for the multi-modal perception pipeline."""
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    lidar_data: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None
    imu_data: Optional[Dict[str, float]] = None
    timestamp: Optional[float] = None
    camera_intrinsics: Optional[Dict[str, float]] = None
    robot_pose: Optional[Dict[str, float]] = None


@dataclass
class PerceptionOutput:
    """Output data from the multi-modal perception pipeline."""
    objects: List[Dict[str, Any]]  # Detected objects with properties
    scene_description: str  # Natural language description of the scene
    spatial_map: Optional[np.ndarray] = None  # 2D/3D spatial representation
    intentions: List[str] = None  # Potential intentions derived from perception
    confidence: float = 0.0  # Overall confidence in the perception
    processing_time: float = 0.0  # Time taken to process the input
    
    def __post_init__(self):
        if self.intentions is None:
            self.intentions = []


class RGBProcessor:
    """Processor for RGB camera data."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image for neural network."""
        if len(image.shape) == 3:
            # Ensure RGB format (some cameras return BGR)
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Apply transforms
        tensor = self.transforms(image_rgb)
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def detect_objects(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        """Detect objects in the RGB image using a pre-trained model."""
        # This is a placeholder - in a real implementation, this would use
        # a pre-trained object detection model like YOLO or DETR
        # For now, we'll return mock detections
        height, width = 480, 640  # Assuming original image size
        
        return [
            {
                "class": "person",
                "confidence": 0.9,
                "bbox": [width * 0.3, height * 0.2, width * 0.7, height * 0.8],  # [x1, y1, x2, y2]
                "center": [(width * 0.3 + width * 0.7) / 2, (height * 0.2 + height * 0.8) / 2]
            },
            {
                "class": "cup",
                "confidence": 0.75,
                "bbox": [width * 0.6, height * 0.3, width * 0.8, height * 0.7],
                "center": [(width * 0.6 + width * 0.8) / 2, (height * 0.3 + height * 0.7) / 2]
            }
        ]


class DepthProcessor:
    """Processor for depth camera data."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def preprocess(self, depth_image: np.ndarray) -> torch.Tensor:
        """Preprocess depth image."""
        # Normalize depth image
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 65535.0  # Normalize 16-bit
        elif depth_image.dtype == np.uint8:
            depth_image = depth_image.astype(np.float32) / 255.0   # Normalize 8-bit
        
        # Add channel dimension
        if len(depth_image.shape) == 2:
            depth_image = np.expand_dims(depth_image, axis=0)
        
        return torch.from_numpy(depth_image).unsqueeze(0).to(self.device)  # Add batch dimension
    
    def extract_surface_normals(self, depth_image: np.ndarray) -> np.ndarray:
        """Extract surface normals from depth image."""
        # Calculate gradients to estimate surface normals
        grad_x = np.gradient(depth_image, axis=1)
        grad_y = np.gradient(depth_image, axis=0)
        
        # Compute normal vectors
        normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        normals[:, :, 0] = -grad_x
        normals[:, :, 1] = -grad_y
        normals[:, :, 2] = 1  # Z component
        
        # Normalize vectors
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        normals = normals / norm
        
        return normals


class LiDARProcessor:
    """Processor for LiDAR point cloud data."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def preprocess(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Preprocess LiDAR point cloud data."""
        # Normalize point cloud to a standard coordinate system
        if point_cloud.shape[1] != 3:
            # Assuming points are in (N, 3) format with x, y, z coordinates
            raise ValueError("Point cloud should have shape (N, 3)")
        
        # Normalize to unit cube for neural network processing
        min_vals = np.min(point_cloud, axis=0)
        max_vals = np.max(point_cloud, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        normalized_points = (point_cloud - min_vals) / range_vals
        
        return torch.from_numpy(normalized_points).unsqueeze(0).to(self.device)
    
    def cluster_objects(self, point_cloud: np.ndarray, eps: float = 0.1, min_samples: int = 10) -> List[np.ndarray]:
        """Cluster points in the point cloud to identify objects."""
        # This is a simplified clustering approach
        # In practice, you might use DBSCAN or other algorithms
        objects = []
        
        # For this example, we'll return mock clusters
        # A real implementation would use proper clustering algorithms
        num_points = point_cloud.shape[0]
        if num_points > 100:
            # Create 3 mock clusters for demonstration
            cluster1 = point_cloud[:num_points//3]
            cluster2 = point_cloud[num_points//3:2*num_points//3]
            cluster3 = point_cloud[2*num_points//3:]
            
            objects = [cluster1, cluster2, cluster3]
        
        return objects


class AudioProcessor:
    """Processor for audio data."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def preprocess(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio data for neural network."""
        # Convert to tensor if needed
        if not isinstance(audio_data, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.float()
        
        # Add batch dimension if needed
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        return audio_tensor.to(self.device)
    
    def extract_features(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio features for perception."""
        # This is a simplified approach - in reality, you'd use
        # mel-spectrograms, MFCCs, or other audio features
        features = {}
        
        # Calculate simple audio features
        features['mean'] = torch.mean(audio_tensor, dim=-1, keepdim=True)
        features['std'] = torch.std(audio_tensor, dim=-1, keepdim=True)
        features['max'] = torch.max(audio_tensor, dim=-1, keepdim=True)[0]
        
        return features


class MultiModalFusion:
    """Fuses information from multiple sensor modalities."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the fusion module."""
        logger = logging.getLogger("MultiModalFusion")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def fuse_features(
        self, 
        rgb_features: Optional[torch.Tensor] = None,
        depth_features: Optional[torch.Tensor] = None,
        lidar_features: Optional[torch.Tensor] = None,
        audio_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Fuse features from multiple modalities."""
        # This is a simplified fusion approach
        # In practice, you'd use more sophisticated methods like attention mechanisms
        
        fused_features = []
        
        if rgb_features is not None:
            fused_features.append(rgb_features)
        
        if depth_features is not None:
            fused_features.append(depth_features)
        
        if lidar_features is not None:
            fused_features.append(lidar_features)
        
        if audio_features is not None:
            # Concatenate all audio features
            audio_concat = torch.cat(list(audio_features.values()), dim=-1)
            fused_features.append(audio_concat)
        
        if not fused_features:
            raise ValueError("At least one modality must be provided")
        
        # Simple concatenation (in practice, this would be more sophisticated)
        if len(fused_features) == 1:
            return fused_features[0]
        
        # Pad tensors to the same size if necessary
        max_size = max(f.shape[-1] for f in fused_features)
        padded_features = []
        
        for feat in fused_features:
            if feat.shape[-1] < max_size:
                padding = max_size - feat.shape[-1]
                padded_feat = torch.nn.functional.pad(feat, (0, padding))
                padded_features.append(padded_feat)
            else:
                padded_features.append(feat)
        
        # Concatenate along the feature dimension
        fused_tensor = torch.cat(padded_features, dim=-1)
        
        return fused_tensor


class MultiModalPerceptionPipeline:
    """Main pipeline that integrates processing from all modalities."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.rgb_processor = RGBProcessor(device)
        self.depth_processor = DepthProcessor(device)
        self.lidar_processor = LiDARProcessor(device)
        self.audio_processor = AudioProcessor(device)
        self.fusion_module = MultiModalFusion(device)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the pipeline."""
        logger = logging.getLogger("MultiModalPerceptionPipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def process(self, input_data: PerceptionInput) -> PerceptionOutput:
        """Process multi-modal input and produce unified perception output."""
        start_time = time.time()
        
        # Process each modality
        rgb_objects = []
        spatial_map = None
        
        if input_data.rgb_image is not None:
            self.logger.info("Processing RGB image")
            rgb_tensor = self.rgb_processor.preprocess(input_data.rgb_image)
            rgb_objects = self.rgb_processor.detect_objects(rgb_tensor)
        
        if input_data.depth_image is not None:
            self.logger.info("Processing depth image")
            depth_tensor = self.depth_processor.preprocess(input_data.depth_image)
            spatial_map = self._create_spatial_map_from_depth(input_data.depth_image)
        
        lidar_objects = []
        if input_data.lidar_data is not None:
            self.logger.info("Processing LiDAR data")
            lidar_tensor = self.lidar_processor.preprocess(input_data.lidar_data)
            lidar_objects = self.lidar_processor.cluster_objects(input_data.lidar_data)
        
        audio_features = None
        if input_data.audio_data is not None:
            self.logger.info("Processing audio data")
            audio_tensor = self.audio_processor.preprocess(input_data.audio_data)
            audio_features = self.audio_processor.extract_features(audio_tensor)
        
        # Fuse information from all modalities
        self.logger.info("Fusing multi-modal information")
        fused_features = self.fusion_module.fuse_features(
            rgb_features=None,  # We use the processed objects instead
            depth_features=None,  # We use the spatial map
            lidar_features=None,  # We use the clustered objects
            audio_features=audio_features
        )
        
        # Generate scene description
        scene_description = self._generate_scene_description(
            rgb_objects, 
            lidar_objects, 
            audio_features is not None
        )
        
        # Determine intentions based on perception
        intentions = self._derive_intentions(rgb_objects, lidar_objects)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(rgb_objects, audio_features)
        
        processing_time = time.time() - start_time
        
        return PerceptionOutput(
            objects=self._merge_objects(rgb_objects, lidar_objects),
            scene_description=scene_description,
            spatial_map=spatial_map,
            intentions=intentions,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _create_spatial_map_from_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Create a 2D spatial map from depth data."""
        # Simple approach: threshold depth to create occupancy grid
        # In practice, this would be more sophisticated
        threshold = np.mean(depth_image) if np.mean(depth_image) > 0 else 1.0
        spatial_map = (depth_image < threshold).astype(np.float32)
        return spatial_map
    
    def _generate_scene_description(
        self, 
        rgb_objects: List[Dict[str, Any]], 
        lidar_objects: List[np.ndarray], 
        has_audio: bool
    ) -> str:
        """Generate a natural language description of the scene."""
        description_parts = []
        
        if rgb_objects:
            object_names = [obj["class"] for obj in rgb_objects]
            unique_objects = list(set(object_names))
            description_parts.append(f"The scene contains {', '.join(unique_objects)}.")
        
        if lidar_objects:
            description_parts.append(f"LiDAR detected {len(lidar_objects)} objects.")
        
        if has_audio:
            description_parts.append("Audio input was received.")
        
        if not description_parts:
            return "The scene appears empty or no objects were detected."
        
        return " ".join(description_parts)
    
    def _derive_intentions(self, rgb_objects: List[Dict[str, Any]], lidar_objects: List[np.ndarray]) -> List[str]:
        """Derive potential intentions from the perceived scene."""
        intentions = []
        
        # If there are known objects, suggest interactions
        for obj in rgb_objects:
            if obj["class"] in ["cup", "bottle", "box"]:
                if obj["confidence"] > 0.7:
                    intentions.append(f"grasp {obj['class']}")
        
        for obj in rgb_objects:
            if obj["class"] in ["person", "human"]:
                if obj["confidence"] > 0.8:
                    intentions.append("greet person")
                    intentions.append("follow person")
        
        # Add navigation intentions if large obstacles detected
        if len(lidar_objects) > 2:
            intentions.append("navigate around obstacles")
        
        return list(set(intentions))  # Remove duplicates
    
    def _calculate_overall_confidence(
        self, 
        rgb_objects: List[Dict[str, Any]], 
        audio_features: Optional[Dict[str, torch.Tensor]]
    ) -> float:
        """Calculate overall confidence in the perception."""
        confidence_parts = []
        
        # Confidence based on number and quality of detected objects
        if rgb_objects:
            avg_obj_confidence = sum(obj["confidence"] for obj in rgb_objects) / len(rgb_objects)
            confidence_parts.append(avg_obj_confidence)
        
        # If we have audio, we can add confidence from that
        if audio_features:
            confidence_parts.append(0.8)  # Assume good audio quality
        
        if not confidence_parts:
            return 0.1  # Low confidence if no data processed
        
        # Return average confidence
        return sum(confidence_parts) / len(confidence_parts)
    
    def _merge_objects(
        self, 
        rgb_objects: List[Dict[str, Any]], 
        lidar_objects: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Merge objects detected from different modalities."""
        merged_objects = []
        
        # Add RGB objects with modality tag
        for obj in rgb_objects:
            obj_copy = obj.copy()
            obj_copy["modality"] = "rgb"
            merged_objects.append(obj_copy)
        
        # Add LiDAR objects with modality tag
        for i, cluster in enumerate(lidar_objects):
            lidar_obj = {
                "class": f"lidar_cluster_{i}",
                "confidence": 0.5,  # Default confidence for LiDAR clusters
                "center": [float(np.mean(cluster[:, 0])), float(np.mean(cluster[:, 1])), float(np.mean(cluster[:, 2]))],
                "modality": "lidar",
                "size": len(cluster)
            }
            merged_objects.append(lidar_obj)
        
        return merged_objects


class AsyncPerceptionPipeline:
    """Asynchronous wrapper for the perception pipeline."""
    
    def __init__(self, device: str = "cuda"):
        self.pipeline = MultiModalPerceptionPipeline(device)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start the asynchronous processing."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the asynchronous processing."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _worker(self):
        """Worker thread for processing."""
        while self.running:
            try:
                input_data = self.input_queue.get(timeout=0.1)
                output = self.pipeline.process(input_data)
                self.output_queue.put(output)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in perception pipeline: {e}")
    
    def submit(self, input_data: PerceptionInput):
        """Submit data for processing."""
        self.input_queue.put(input_data)
    
    def get_result(self, timeout: float = 1.0) -> Optional[PerceptionOutput]:
        """Get the latest result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def example_usage():
    """Example of how to use the multi-modal perception pipeline."""
    print("Multi-modal Perception Pipeline Example")
    print("=" * 50)
    
    # Create a perception pipeline
    pipeline = MultiModalPerceptionPipeline()
    
    # Create example input data
    # RGB image (simulated)
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Depth image (simulated)
    depth_image = np.random.rand(480, 640).astype(np.float32) * 5.0  # meters
    
    # LiDAR data (simulated point cloud)
    lidar_points = np.random.rand(1000, 3).astype(np.float32)  # 1000 points in 3D space
    
    # Audio data (simulated)
    audio_data = np.random.rand(16000).astype(np.float32)  # 1 second at 16kHz
    
    # Create input
    input_data = PerceptionInput(
        rgb_image=rgb_image,
        depth_image=depth_image,
        lidar_data=lidar_points,
        audio_data=audio_data,
        timestamp=time.time()
    )
    
    # Process the input
    print("Processing multi-modal input...")
    result = pipeline.process(input_data)
    
    print(f"Processing time: {result.processing_time:.3f} seconds")
    print(f"Overall confidence: {result.confidence:.2f}")
    print(f"Scene description: {result.scene_description}")
    print(f"Detected objects: {len(result.objects)}")
    print(f"Derived intentions: {result.intentions}")
    
    # Example with asynchronous pipeline
    print("\nTesting asynchronous pipeline...")
    async_pipeline = AsyncPerceptionPipeline()
    async_pipeline.start()
    
    # Submit the same input
    async_pipeline.submit(input_data)
    
    # Get result
    result_async = async_pipeline.get_result(timeout=5.0)
    if result_async:
        print(f"Asynchronous processing time: {result_async.processing_time:.3f} seconds")
        print(f"Asynchronous result confidence: {result_async.confidence:.2f}")
    else:
        print("No result returned from async pipeline")
    
    # Stop the async pipeline
    async_pipeline.stop()


if __name__ == "__main__":
    example_usage()