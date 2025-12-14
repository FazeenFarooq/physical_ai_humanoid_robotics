"""
Real-time Inference Examples for the Physical AI & Humanoid Robotics Course.

This module provides examples of real-time inference implementations optimized 
for NVIDIA Jetson Orin hardware, following the course's emphasis on GPU-accelerated 
perception and real-time operation.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import threading
from collections import deque
import queue
from dataclasses import dataclass
import logging
import subprocess


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""
    model_path: str
    input_shape: Tuple[int, int, int]  # (channels, height, width)
    batch_size: int = 1
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    device: str = "cuda"  # "cuda" or "cpu"
    precision: str = "fp16"  # "fp32", "fp16", or "int8"
    max_buffer_size: int = 30  # Maximum frames in buffer
    target_fps: float = 30.0  # Target processing rate


class RealTimeInferenceEngine:
    """Base class for real-time inference engines."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.transform = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=config.max_buffer_size)
        self.result_queue = queue.Queue(maxsize=config.max_buffer_size)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the inference engine."""
        logger = logging.getLogger(f"RealTimeInference-{self.config.model_path.split('/')[-1]}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_model(self):
        """Load and prepare the model for inference."""
        raise NotImplementedError("Subclasses must implement load_model method")
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess input frame for model inference."""
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def postprocess(self, outputs: Any, original_shape: Tuple[int, int]) -> Any:
        """Postprocess model outputs."""
        raise NotImplementedError("Subclasses must implement postprocess method")
    
    def inference(self, input_tensor: torch.Tensor) -> Any:
        """Run inference on the input tensor."""
        with torch.no_grad():
            if self.config.precision == "fp16":
                with torch.cuda.amp.autocast():
                    return self.model(input_tensor)
            else:
                return self.model(input_tensor)
    
    def start_inference_pipeline(self):
        """Start the real-time inference pipeline in a separate thread."""
        self.is_running = True
        
        # Start frame processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.start()
        
        self.logger.info("Real-time inference pipeline started")
    
    def stop_inference_pipeline(self):
        """Stop the real-time inference pipeline."""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        self.logger.info("Real-time inference pipeline stopped")
    
    def _process_frames(self):
        """Process frames from the queue."""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Preprocess frame
                input_tensor = self.preprocess(frame)
                
                # Run inference
                start_time = time.time()
                outputs = self.inference(input_tensor)
                inference_time = time.time() - start_time
                
                # Postprocess outputs
                original_shape = frame.shape[:2]
                results = self.postprocess(outputs, original_shape)
                
                # Add timing information
                results['inference_time'] = inference_time
                results['timestamp'] = time.time()
                
                # Put results in output queue
                try:
                    self.result_queue.put_nowait(results)
                except queue.Full:
                    # If result queue is full, we lose this result
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
    
    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit a frame for inference processing."""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            self.logger.warning("Frame queue is full, dropping frame")
            return False
    
    def get_latest_results(self) -> Optional[Any]:
        """Get the latest inference results."""
        results = None
        try:
            # Get the latest result, discarding older ones
            while True:
                results = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        return results


class ObjectDetectionInference(RealTimeInferenceEngine):
    """Real-time object detection inference engine."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.input_shape[1], self.config.input_shape[2])),
            transforms.ToTensor(),
        ])
    
    def load_model(self):
        """Load a pre-trained object detection model."""
        # In a real implementation, we would load models like YOLO, SSD, etc.
        # For this example, we'll create a mock model
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1000)  # 1000 classes as an example
        ).to(self.config.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        self.logger.info(f"Loaded object detection model on {self.config.device}")
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for object detection."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to model input size
        frame_resized = cv2.resize(frame_rgb, (self.config.input_shape[2], self.config.input_shape[1]))
        
        # Convert to tensor and normalize
        input_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.config.device) / 255.0  # Normalize to [0, 1]
        
        return input_tensor
    
    def postprocess(self, outputs: torch.Tensor, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Postprocess detection outputs."""
        # For this example, return mock detection results
        # In a real implementation, this would decode the actual model outputs
        
        # Get the original frame height and width
        orig_h, orig_w = original_shape
        
        # Mock detections - in a real implementation, this would be actual detection results
        detections = []
        
        # Convert outputs to probabilities
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k=5)
        
        for i in range(top_probs.shape[1]):
            class_id = top_classes[0, i].item()
            confidence = top_probs[0, i].item()
            
            if confidence > self.config.confidence_threshold:
                # Mock bounding box (in a real implementation, this would come from the model)
                bbox = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [0.1 * orig_w, 0.1 * orig_h, 0.3 * orig_w, 0.2 * orig_h]  # [x1, y1, x2, y2]
                }
                detections.append(bbox)
        
        return {
            'detections': detections,
            'original_shape': original_shape,
            'status': 'success'
        }


class VisionLanguageActionInference(RealTimeInferenceEngine):
    """Real-time VLA (Vision-Language-Action) inference engine."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        
        # Initialize common transforms
        self.vision_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.input_shape[1], self.config.input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load a VLA model (vision-language-action model)."""
        # For this example, we'll create a mock VLA model
        # In a real implementation, this would be a transformer-based model
        # that processes visual input and produces action commands
        
        # Mock vision encoder
        self.vision_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 512)
        ).to(self.config.device)
        
        # Mock language model (simplified)
        self.language_model = torch.nn.Linear(512, 256).to(self.config.device)
        
        # Mock action head
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),  # 64 action parameters
        ).to(self.config.device)
        
        self.vision_encoder.eval()
        self.language_model.eval()
        self.action_head.eval()
        
        self.logger.info(f"Loaded VLA model on {self.config.device}")
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for VLA model."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to model input size
        frame_resized = cv2.resize(frame_rgb, (self.config.input_shape[2], self.config.input_shape[1]))
        
        # Convert to tensor and normalize
        input_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1)
        input_tensor = input_tensor.to(self.config.device)
        
        return input_tensor.unsqueeze(0)  # Add batch dimension
    
    def postprocess(self, outputs: torch.Tensor, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Postprocess VLA outputs."""
        # In a real implementation, this would decode the action commands
        # For this example, return mock action results
        
        # Mock action parameters (in a real implementation, these would be actual action values)
        action_params = {
            'linear_velocity': float(outputs[0, 0].cpu()) if outputs.numel() > 0 else 0.0,
            'angular_velocity': float(outputs[0, 1].cpu()) if outputs.numel() > 1 else 0.0,
            'gripper_position': float(outputs[0, 2].cpu()) if outputs.numel() > 2 else 0.5,
            'target_object_class': 'unknown',
            'confidence': 0.8
        }
        
        return {
            'action_params': action_params,
            'original_shape': original_shape,
            'status': 'success'
        }
    
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run VLA inference on the input tensor."""
        with torch.no_grad():
            # Run vision encoder
            if self.config.precision == "fp16":
                with torch.cuda.amp.autocast():
                    vision_features = self.vision_encoder(input_tensor)
            else:
                vision_features = self.vision_encoder(input_tensor)
            
            # Process with language model (simplified)
            lang_features = self.language_model(vision_features)
            
            # Combine and generate action
            combined_features = torch.cat([vision_features, lang_features], dim=1)
            
            if self.config.precision == "fp16":
                with torch.cuda.amp.autocast():
                    actions = self.action_head(combined_features)
            else:
                actions = self.action_head(combined_features)
        
        return actions


class RealTimeInferenceMonitor:
    """Monitor for real-time inference performance."""
    
    def __init__(self, window_size: int = 100):
        self.fps_history = deque(maxlen=window_size)
        self.inference_time_history = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()
        self.logger = logging.getLogger("InferenceMonitor")
    
    def update(self, inference_time: float):
        """Update with new inference timing data."""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
        else:
            fps = 0
        
        # Store metrics
        self.fps_history.append(fps)
        self.inference_time_history.append(inference_time * 1000)  # Convert to milliseconds
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.fps_history:
            return {
                'current_fps': 0.0,
                'avg_fps': 0.0,
                'min_fps': 0.0,
                'max_fps': 0.0,
                'avg_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0
            }
        
        return {
            'current_fps': self.fps_history[-1],
            'avg_fps': np.mean(self.fps_history),
            'min_fps': min(self.fps_history),
            'max_fps': max(self.fps_history),
            'avg_inference_time_ms': np.mean(self.inference_time_history),
            'min_inference_time_ms': min(self.inference_time_history),
            'max_inference_time_ms': max(self.inference_time_history)
        }
    
    def print_performance(self):
        """Print current performance metrics."""
        metrics = self.get_performance_metrics()
        print(f"FPS: {metrics['avg_fps']:.2f} (avg), {metrics['current_fps']:.2f} (current)")
        print(f"Inference: {metrics['avg_inference_time_ms']:.2f}ms (avg)")


def run_object_detection_example():
    """Run a real-time object detection example."""
    print("Running real-time object detection example...")
    
    # Configuration for object detection
    config = InferenceConfig(
        model_path="yolo_v5",  # This would be the actual model path
        input_shape=(3, 416, 416),  # YOLO input size
        confidence_threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create inference engine
    detector = ObjectDetectionInference(config)
    detector.load_model()
    
    # Create performance monitor
    monitor = RealTimeInferenceMonitor()
    
    # Start inference pipeline
    detector.start_inference_pipeline()
    
    # Simulate real-time input (in a real scenario, this would be from a camera)
    print("Processing frames (simulated)...")
    
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated frame
    
    try:
        for i in range(100):  # Process 100 frames
            # Submit frame
            detector.submit_frame(test_frame)
            
            # Get results
            results = detector.get_latest_results()
            if results:
                # Update performance monitor
                monitor.update(results.get('inference_time', 0))
                
                # Print metrics periodically
                if i % 20 == 0:
                    print(f"Processed frame {i}")
                    monitor.print_performance()
            
            # Simulate real-time capture rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Stop inference pipeline
    detector.stop_inference_pipeline()
    
    print("Object detection example completed")


def run_vla_example():
    """Run a real-time Vision-Language-Action example."""
    print("\nRunning real-time Vision-Language-Action example...")
    
    # Configuration for VLA
    config = InferenceConfig(
        model_path="vla_model",  # This would be the actual model path
        input_shape=(3, 224, 224),  # Standard Vision Transformer input size
        confidence_threshold=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create inference engine
    vla_engine = VisionLanguageActionInference(config)
    vla_engine.load_model()
    
    # Create performance monitor
    monitor = RealTimeInferenceMonitor()
    
    # Start inference pipeline
    vla_engine.start_inference_pipeline()
    
    # Simulate real-time input
    print("Processing VLA frames (simulated)...")
    
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated frame
    
    try:
        for i in range(50):  # Process 50 frames
            # Submit frame
            vla_engine.submit_frame(test_frame)
            
            # Get results
            results = vla_engine.get_latest_results()
            if results:
                # Update performance monitor
                monitor.update(results.get('inference_time', 0))
                
                # Print metrics periodically
                if i % 10 == 0:
                    print(f"Processed VLA frame {i}")
                    monitor.print_performance()
                    print(f"Action params: {results.get('action_params', {})}")
            
            # Simulate real-time capture rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Stop inference pipeline
    vla_engine.stop_inference_pipeline()
    
    print("VLA example completed")


def check_jetson_performance():
    """Check performance capabilities of the Jetson platform."""
    print("\nChecking Jetson performance capabilities...")
    
    # Check GPU information
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GPU Info: {result.stdout.strip()}")
    except Exception as e:
        print(f"Could not get GPU info: {e}")
    
    # Check available memory
    try:
        result = subprocess.run(['cat', '/proc/meminfo'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'MemAvailable' in line or 'MemTotal' in line:
                    print(f"{line.strip()}")
    except Exception as e:
        print(f"Could not get memory info: {e}")
    
    # Estimate theoretical performance
    # This is a simplified estimation based on GPU capabilities
    print("\nEstimated Performance for Jetson Orin:")
    print("- FP32 Tensor Core Performance: ~275 TOPS")
    print("- FP16 Tensor Core Performance: ~1370 TOPS") 
    print("- INT8 Tensor Core Performance: ~2740 TOPS")
    print("- Memory Bandwidth: ~204.8 GB/s")


if __name__ == "__main__":
    # Run examples
    check_jetson_performance()
    run_object_detection_example()
    run_vla_example()
    
    print("\nReal-time inference examples completed!")