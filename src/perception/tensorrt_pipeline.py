"""
TensorRT Optimization Pipeline for the Physical AI & Humanoid Robotics Course.

This module provides tools for optimizing deep learning models using NVIDIA TensorRT,
enabling real-time inference on Jetson Orin hardware as required by the course's 
emphasis on GPU-accelerated perception and training.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import torch_tensorrt
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
from pathlib import Path
import cv2
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for TensorRT optimization."""
    precision: str = "fp16"  # fp32, fp16, int8
    max_workspace_size: int = 1 << 30  # 1GB
    min_timing_iterations: int = 2
    avg_timing_iterations: int = 1
    max_batch_size: int = 1
    dynamic_shapes: Optional[Dict[str, Tuple[int, int, int]]] = None
    calibration_dataset: Optional[str] = None  # For INT8 calibration
    engine_save_path: str = ""
    verbose: bool = False


class TensorRTOptimizer:
    """Optimizer for converting and optimizing models with TensorRT."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE if self.logger.level <= 10 else trt.Logger.WARNING)
    
    def _setup_logger(self):
        """Set up a basic logger."""
        import logging
        logger = logging.getLogger("TensorRTOptimizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def optimize_pytorch_model(
        self, 
        model: torch.nn.Module, 
        config: OptimizationConfig,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> trt.ICudaEngine:
        """
        Optimize a PyTorch model using TensorRT.
        
        Args:
            model: PyTorch model to optimize
            config: Optimization configuration
            example_inputs: Example inputs for tracing the model
            
        Returns:
            TensorRT engine
        """
        try:
            self.logger.info("Starting PyTorch model optimization with TensorRT")
            
            # Set model to evaluation mode
            model.eval()
            
            # Convert the model to TorchScript
            self.logger.info("Converting PyTorch model to TorchScript")
            traced_model = torch.jit.trace(model, example_inputs)
            traced_model.eval()
            
            # Prepare optimization settings
            trt_settings = {
                "inputs": [torch_tensorrt.Input(
                    shape=example_inputs.shape if isinstance(example_inputs, torch.Tensor) 
                    else example_inputs[0].shape,
                    dtype=torch.float
                )],
                "enabled_precisions": {getattr(torch_tensorrt.dtype, config.precision.upper())}
                if config.precision.upper() in ["FP32", "FP16", "INT8"] 
                else {torch_tensorrt.dtype.FP16},
                "workspace_size": config.max_workspace_size,
                "max_batch_size": config.max_batch_size,
            }
            
            # Compile the model with TensorRT
            self.logger.info(f"Compiling model with TensorRT using {config.precision} precision")
            trt_compiled_module = torch_tensorrt.compile(
                traced_model,
                **trt_settings
            )
            
            # Save the optimized engine
            if config.engine_save_path:
                with open(config.engine_save_path, "wb") as f:
                    f.write(trt_compiled_module._c._module._get_method("forward").engine_bytes())
                self.logger.info(f"Optimized engine saved to {config.engine_save_path}")
            
            self.logger.info("PyTorch model optimization completed successfully")
            return trt_compiled_module._c._module._get_method("forward").engine_bytes()
            
        except Exception as e:
            self.logger.error(f"Error during PyTorch model optimization: {e}")
            raise
    
    def create_trt_engine_from_onnx(
        self, 
        onnx_model_path: str, 
        config: OptimizationConfig
    ) -> trt.ICudaEngine:
        """
        Create a TensorRT engine from an ONNX model.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            config: Optimization configuration
            
        Returns:
            TensorRT engine
        """
        try:
            self.logger.info(f"Creating TensorRT engine from ONNX model: {onnx_model_path}")
            
            # Create builder
            builder = trt.Builder(self.trt_logger)
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create parser
            parser = trt.OnnxParser(network, self.trt_logger)
            
            # Parse ONNX model
            with open(onnx_model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config_builder = builder.create_builder_config()
            config_builder.max_workspace_size = config.max_workspace_size
            
            # Set precision
            if config.precision.lower() == "fp16":
                if builder.platform_has_fast_fp16:
                    config_builder.set_flag(trt.BuilderFlag.FP16)
                    self.logger.info("Using FP16 precision")
                else:
                    self.logger.warning("Platform does not support FP16, using FP32")
            elif config.precision.lower() == "int8":
                if builder.platform_has_fast_int8:
                    config_builder.set_flag(trt.BuilderFlag.INT8)
                    self.logger.info("Using INT8 precision")
                    
                    # Set up calibration if dataset provided
                    if config.calibration_dataset:
                        self.logger.info("Setting up INT8 calibration")
                        # Calibration implementation would go here
                else:
                    self.logger.warning("Platform does not support INT8, using FP32")
            
            # Build engine
            self.logger.info("Building TensorRT engine...")
            start_time = time.time()
            engine = builder.build_engine(network, config_builder)
            build_time = time.time() - start_time
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            self.logger.info(f"TensorRT engine built successfully in {build_time:.2f} seconds")
            
            # Save engine if path specified
            if config.engine_save_path:
                with open(config.engine_save_path, "wb") as f:
                    f.write(engine.serialize())
                self.logger.info(f"TensorRT engine saved to {config.engine_save_path}")
            
            return engine
            
        except Exception as e:
            self.logger.error(f"Error creating TensorRT engine from ONNX: {e}")
            raise
    
    def benchmark_model(
        self, 
        engine: trt.ICudaEngine, 
        input_shape: Tuple[int, ...], 
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark the performance of a TensorRT engine.
        
        Args:
            engine: TensorRT engine to benchmark
            input_shape: Shape of the input tensor
            num_runs: Number of inference runs for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            self.logger.info(f"Benchmarking TensorRT engine with {num_runs} runs")
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate buffers
            inputs, outputs, bindings, stream = self._allocate_buffers(engine, input_shape)
            
            # Generate random input data
            input_data = np.random.random(input_shape).astype(np.float32)
            
            # Warm-up runs
            for _ in range(10):
                self._do_inference(context, bindings, input_data, stream)
            
            # Timing runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                self._do_inference(context, bindings, input_data, stream)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            
            metrics = {
                "avg_inference_time_ms": avg_time,
                "std_inference_time_ms": std_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "fps": fps,
                "num_runs": num_runs
            }
            
            self.logger.info(f"Benchmark completed: {fps:.2f} FPS, avg: {avg_time:.2f}ms")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {e}")
            raise
    
    def _allocate_buffers(self, engine: trt.ICudaEngine, input_shape: Tuple[int, ...]):
        """Allocate input and output buffers for TensorRT inference."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for idx in range(engine.num_bindings):
            binding_shape = engine.get_binding_shape(idx)
            size = trt.volume(binding_shape) * engine.max_batch_size * 4  # 4 bytes per float32
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(trt.volume(binding_shape) * engine.max_batch_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(idx):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})
        
        return inputs, outputs, bindings, stream
    
    def _do_inference(self, context, bindings, input_data, stream):
        """Perform a single inference run."""
        # Transfer input data to device
        np.copyto(bindings[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(bindings[0]["device"], bindings[0]["host"], stream)
        
        # Execute inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(bindings[1]["host"], bindings[1]["device"], stream)
        
        # Synchronize stream
        stream.synchronize()


class PerceptionModelOptimizer:
    """High-level optimizer for perception models in robotics applications."""
    
    def __init__(self):
        self.trt_optimizer = TensorRTOptimizer()
        self.logger = self.trt_optimizer.logger
    
    def optimize_detection_model(
        self, 
        model_path: str, 
        config: OptimizationConfig,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Tuple[trt.ICudaEngine, Dict[str, float]]:
        """
        Optimize an object detection model for robotics perception.
        
        Args:
            model_path: Path to the model (ONNX format)
            config: Optimization configuration
            input_shape: Shape of the input tensor
            
        Returns:
            Tuple of optimized engine and benchmark metrics
        """
        self.logger.info("Optimizing object detection model")
        
        # Load and optimize the model
        engine = self.trt_optimizer.create_trt_engine_from_onnx(model_path, config)
        
        # Benchmark the optimized model
        metrics = self.trt_optimizer.benchmark_model(engine, input_shape, num_runs=50)
        
        self.logger.info("Object detection model optimization completed")
        return engine, metrics
    
    def optimize_vla_model(
        self, 
        model_path: str, 
        config: OptimizationConfig,
        input_shapes: Dict[str, Tuple[int, ...]],
        model_type: str = "vision_language_action"
    ) -> Tuple[trt.ICudaEngine, Dict[str, float]]:
        """
        Optimize a Vision-Language-Action model for integrated robotics tasks.
        
        Args:
            model_path: Path to the model (ONNX format)
            config: Optimization configuration
            input_shapes: Dictionary mapping input names to shapes
            model_type: Type of VLA model
            
        Returns:
            Tuple of optimized engine and benchmark metrics
        """
        self.logger.info(f"Optimizing {model_type} model")
        
        # For this example, we'll use a single input shape, but a real implementation
        # would handle multiple inputs for VLA models
        first_shape = next(iter(input_shapes.values()))
        engine = self.trt_optimizer.create_trt_engine_from_onnx(model_path, config)
        
        # Benchmark the optimized model
        metrics = self.trt_optimizer.benchmark_model(engine, first_shape, num_runs=30)
        
        self.logger.info(f"{model_type} model optimization completed")
        return engine, metrics


def create_optimization_config_from_dict(config_dict: Dict[str, Any]) -> OptimizationConfig:
    """Create an OptimizationConfig from a dictionary."""
    return OptimizationConfig(
        precision=config_dict.get("precision", "fp16"),
        max_workspace_size=config_dict.get("max_workspace_size", 1 << 30),
        min_timing_iterations=config_dict.get("min_timing_iterations", 2),
        avg_timing_iterations=config_dict.get("avg_timing_iterations", 1),
        max_batch_size=config_dict.get("max_batch_size", 1),
        dynamic_shapes=config_dict.get("dynamic_shapes"),
        calibration_dataset=config_dict.get("calibration_dataset"),
        engine_save_path=config_dict.get("engine_save_path", ""),
        verbose=config_dict.get("verbose", False)
    )


def example_usage():
    """Example of how to use the TensorRT optimization pipeline."""
    # Create optimizer instance
    optimizer = PerceptionModelOptimizer()
    
    # Example optimization configuration
    config = OptimizationConfig(
        precision="fp16",  # Use FP16 for balance of accuracy and performance
        max_workspace_size=1 << 30,  # 1GB maximum workspace
        max_batch_size=1,  # Single image inference for robotics
        engine_save_path="./optimized_model.engine"
    )
    
    # Example for optimizing a detection model
    try:
        engine, metrics = optimizer.optimize_detection_model(
            model_path="./models/detection_model.onnx",
            config=config,
            input_shape=(1, 3, 416, 416)  # YOLO input size
        )
        
        print("Optimization completed!")
        print(f"Performance: {metrics['fps']:.2f} FPS, {metrics['avg_inference_time_ms']:.2f} ms avg")
        
        # Save optimization report
        report = {
            "optimization_config": {
                "precision": config.precision,
                "workspace_size": config.max_workspace_size,
                "batch_size": config.max_batch_size
            },
            "performance_metrics": metrics
        }
        
        with open("optimization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("Optimization report saved to optimization_report.json")
        
    except Exception as e:
        print(f"Optimization failed: {e}")


if __name__ == "__main__":
    example_usage()