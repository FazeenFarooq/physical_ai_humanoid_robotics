"""
TensorRT Optimization Examples for Robotics Perception

This module provides examples and utilities for optimizing perception models 
using NVIDIA TensorRT for deployment on edge computing platforms like Jetson Orin.
"""

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import logging
import time
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """
    Optimizes PyTorch models using TensorRT for faster inference on NVIDIA hardware
    """
    
    def __init__(self, precision: str = "fp16"):
        """
        Initialize the TensorRT optimizer
        
        Args:
            precision: Precision mode ("fp32", "fp16", "int8")
        """
        self.precision = precision
        self.logger = logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
    def optimize_model(self, 
                      model: torch.nn.Module, 
                      input_shape: Tuple[int, ...], 
                      output_path: str) -> str:
        """
        Optimize a PyTorch model using TensorRT
        
        Args:
            model: PyTorch model to optimize
            input_shape: Shape of the input tensor (e.g., (1, 3, 224, 224))
            output_path: Path to save the optimized TensorRT engine
            
        Returns:
            Path to the optimized engine file
        """
        logger.info(f"Optimizing model with input shape {input_shape}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create a dummy input to trace the model
        dummy_input = torch.randn(input_shape)
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to ONNX
        onnx_path = output_path.replace('.plan', '.onnx')
        torch.onnx.export(
            traced_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Build TensorRT engine from ONNX
        engine_path = self._build_engine_from_onnx(onnx_path, input_shape, output_path)
        
        # Clean up ONNX file
        os.remove(onnx_path)
        
        logger.info(f"Model optimized and saved to {engine_path}")
        return engine_path
    
    def _build_engine_from_onnx(self, 
                               onnx_path: str, 
                               input_shape: Tuple[int, ...], 
                               output_path: str) -> str:
        """
        Build a TensorRT engine from an ONNX file
        """
        # Create builder and network
        with trt.Builder(self.trt_logger) as builder, \
             builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             builder.create_builder_config() as config, \
             trt.OnnxParser(network, self.trt_logger) as parser:
            
            # Parse ONNX
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX file")
            
            # Configure optimization settings
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Set precision
            if self.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    logger.warning("FP16 not supported on this platform, using FP32")
            elif self.precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    # For INT8, we would need calibration data
                    # config.int8_calibrator = YourCalibrator()
                else:
                    logger.warning("INT8 not supported on this platform, using FP32")
            
            # Set input shape
            profile = builder.create_optimization_profile()
            profile.set_shape("input", input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            return output_path


class TRTInferenceEngine:
    """
    Inference engine for running TensorRT optimized models
    """
    
    def __init__(self, engine_path: str):
        """
        Initialize the inference engine with a TensorRT engine file
        
        Args:
            engine_path: Path to the TensorRT engine file (.plan)
        """
        self.engine_path = engine_path
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """
        Load a TensorRT engine from file
        """
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        return engine
    
    def _allocate_buffers(self):
        """
        Allocate input and output buffers for the TensorRT engine
        """
        # Get input and output binding information
        self.input_binding_idx = None
        self.output_binding_idx = None
        
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_binding_idx = idx
                self.input_shape = self.engine.get_binding_shape(idx)
                self.input_size = trt.volume(self.input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
            else:
                self.output_binding_idx = idx
                self.output_shape = self.engine.get_binding_shape(idx)
                self.output_size = trt.volume(self.output_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
        
        if self.input_binding_idx is None or self.output_binding_idx is None:
            raise RuntimeError("Could not find input or output binding")
        
        # Allocate CUDA memory
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        # Create a stream for async execution
        self.stream = cuda.Stream()
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform inference on input data
        
        Args:
            input_data: Input data as numpy array, shape should match engine input
            
        Returns:
            Output data as numpy array
        """
        # Allocate output buffer
        output_buffer = np.empty(self.output_shape, dtype=np.float32)
        
        # Copy input data to GPU memory
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Execute inference
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        
        # Copy output data back to CPU memory
        cuda.memcpy_dtoh_async(output_buffer, self.d_output, self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        return output_buffer


class PerceptionModelOptimizer:
    """
    Specialized optimizer for perception models used in robotics
    """
    
    def __init__(self):
        self.tensorrt_optimizer = TensorRTOptimizer()
    
    def optimize_object_detection_model(self, 
                                      model: torch.nn.Module, 
                                      input_shape: Tuple[int, ...], 
                                      output_path: str) -> str:
        """
        Optimize an object detection model specifically
        """
        logger.info("Optimizing object detection model...")
        
        # Perform standard optimization
        engine_path = self.tensorrt_optimizer.optimize_model(model, input_shape, output_path)
        
        # Additional post-processing for object detection models
        self._post_process_detection_model(engine_path)
        
        return engine_path
    
    def _post_process_detection_model(self, engine_path: str):
        """
        Apply detection-specific optimizations
        """
        logger.info(f"Applying post-processing optimizations to {engine_path}")
        # Add any detection-specific optimizations here
        # For example, optimizing NMS operations or confidence thresholding
    
    def optimize_segmentation_model(self, 
                                  model: torch.nn.Module, 
                                  input_shape: Tuple[int, ...], 
                                  output_path: str) -> str:
        """
        Optimize a semantic segmentation model specifically
        """
        logger.info("Optimizing semantic segmentation model...")
        
        # Perform standard optimization
        return self.tensorrt_optimizer.optimize_model(model, input_shape, output_path)
    
    def optimize_depth_estimation_model(self, 
                                      model: torch.nn.Module, 
                                      input_shape: Tuple[int, ...], 
                                      output_path: str) -> str:
        """
        Optimize a depth estimation model specifically
        """
        logger.info("Optimizing depth estimation model...")
        
        # Perform standard optimization
        return self.tensorrt_optimizer.optimize_model(model, input_shape, output_path)


def benchmark_model_performance(model_path: str, 
                              input_shape: Tuple[int, ...], 
                              num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark the performance of a TensorRT optimized model
    
    Args:
        model_path: Path to the TensorRT engine file
        input_shape: Shape of the input tensor
        num_runs: Number of inference runs for benchmarking
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Benchmarking model: {model_path}")
    
    # Create inference engine
    engine = TRTInferenceEngine(model_path)
    
    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    input_data = np.ascontiguousarray(input_data)
    
    # Warm up
    for _ in range(10):
        _ = engine.infer(input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = engine.infer(input_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    metrics = {
        'total_time': total_time,
        'num_runs': num_runs,
        'avg_time_per_inference': avg_time,
        'fps': fps,
        'throughput': num_runs / total_time
    }
    
    logger.info(f"Performance metrics: {metrics}")
    return metrics


def optimize_perception_pipeline(model_configs: List[Dict[str, Any]], 
                                output_dir: str) -> List[str]:
    """
    Optimize an entire perception pipeline with multiple models
    
    Args:
        model_configs: List of dictionaries containing model config:
                      {
                        'model': torch.nn.Module,
                        'input_shape': Tuple[int, ...],
                        'name': str,
                        'task': str  # 'detection', 'segmentation', 'depth', etc.
                      }
        output_dir: Directory to save optimized models
        
    Returns:
        List of paths to optimized engine files
    """
    logger.info(f"Optimizing perception pipeline with {len(model_configs)} models")
    
    optimizer = PerceptionModelOptimizer()
    output_paths = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(model_configs):
        model = config['model']
        input_shape = config['input_shape']
        name = config['name']
        task = config.get('task', 'general')
        
        engine_output_path = str(output_path / f"{name}.plan")
        
        if task == 'detection':
            engine_path = optimizer.optimize_object_detection_model(
                model, input_shape, engine_output_path
            )
        elif task == 'segmentation':
            engine_path = optimizer.optimize_segmentation_model(
                model, input_shape, engine_output_path
            )
        elif task == 'depth':
            engine_path = optimizer.optimize_depth_estimation_model(
                model, input_shape, engine_output_path
            )
        else:
            # General optimization
            engine_path = optimizer.tensorrt_optimizer.optimize_model(
                model, input_shape, engine_output_path
            )
        
        output_paths.append(engine_path)
        logger.info(f"Optimized model {i+1}/{len(model_configs)}: {name}")
    
    logger.info(f"Pipeline optimization complete. Models saved to {output_dir}")
    return output_paths


# Example usage functions
def example_optimize_simple_model():
    """
    Example of optimizing a simple PyTorch model
    """
    logger.info("Running example: optimizing a simple model")
    
    # Define a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
            self.fc1 = torch.nn.Linear(64 * 62 * 62, 128)
            self.fc2 = torch.nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    input_shape = (1, 3, 256, 256)
    
    optimizer = TensorRTOptimizer()
    engine_path = optimizer.optimize_model(
        model, 
        input_shape, 
        "simple_model.plan"
    )
    
    # Run inference with the optimized model
    engine = TRTInferenceEngine(engine_path)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    dummy_input = np.ascontiguousarray(dummy_input)
    
    result = engine.infer(dummy_input)
    logger.info(f"Inference result shape: {result.shape}")
    
    # Clean up
    if os.path.exists(engine_path):
        os.remove(engine_path)
    
    logger.info("Simple model optimization example completed")


def example_optimize_detection_model():
    """
    Example of optimizing a detection model
    """
    logger.info("Running example: optimizing a detection model")
    
    # Using a pre-trained model for demonstration
    try:
        import torchvision
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        
        input_shape = (1, 3, 416, 416)
        
        optimizer = PerceptionModelOptimizer()
        engine_path = optimizer.optimize_object_detection_model(
            model,
            input_shape,
            "detection_model.plan"
        )
        
        # Benchmark the optimized model
        metrics = benchmark_model_performance(engine_path, input_shape, num_runs=50)
        logger.info(f"Detection model metrics: {metrics}")
        
        # Clean up
        if os.path.exists(engine_path):
            os.remove(engine_path)
        
        logger.info("Detection model optimization example completed")
    except ImportError:
        logger.warning("torchvision not available, skipping detection model example")


def main():
    """
    Main function to demonstrate TensorRT optimization examples
    """
    logger.info("Starting TensorRT optimization examples")
    
    # Run examples
    example_optimize_simple_model()
    example_optimize_detection_model()
    
    logger.info("All TensorRT optimization examples completed")


if __name__ == "__main__":
    main()