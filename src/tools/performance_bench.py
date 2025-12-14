"""
Performance Benchmarking Tools for the Physical AI & Humanoid Robotics Course.

This module provides comprehensive tools for benchmarking the performance of AI models
and robotics applications, particularly for deployment on resource-constrained platforms
like the NVIDIA Jetson Orin.
"""

import time
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
import json
import csv
import threading
import queue
import statistics
from dataclasses import dataclass
from pathlib import Path
import logging
from enum import Enum


class BenchmarkType(Enum):
    """Type of benchmark to run."""
    INFERENCE = "inference"
    END_TO_END = "end_to_end"
    MEMORY = "memory"
    POWER = "power"
    THROUGHPUT = "throughput"


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    benchmark_type: BenchmarkType
    model_path: str
    input_shapes: List[Tuple[int, ...]]
    num_runs: int = 100
    warmup_runs: int = 10
    batch_size: int = 1
    device: str = "cuda"
    precision: str = "fp32"  # fp32, fp16, int8
    save_results: bool = True
    results_path: str = "benchmark_results/"
    include_memory: bool = False
    include_power: bool = False
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_type: BenchmarkType
    timestamp: float
    avg_inference_time_ms: float
    std_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    avg_fps: float
    memory_usage_mb: Optional[float] = None
    power_draw_w: Optional[float] = None
    throughput_fps: Optional[float] = None
    input_shapes: Optional[List[Tuple[int, ...]]] = None
    num_runs: Optional[int] = None
    model_size_mb: Optional[float] = None


class PerformanceBenchmark:
    """Base class for performance benchmarking."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results: List[BenchmarkResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the benchmark."""
        logger = logging.getLogger(f"PerformanceBenchmark-{self.config.benchmark_type.value}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        raise NotImplementedError("Subclasses must implement run_benchmark method")
    
    def _run_inference_benchmark(self, model: Any, example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        """Run inference benchmark."""
        # Move model to device
        model.to(self.config.device)
        model.eval()
        
        # Prepare inputs
        if isinstance(example_inputs, torch.Tensor):
            inputs = example_inputs.to(self.config.device)
        else:
            inputs = tuple(inp.to(self.config.device) for inp in example_inputs)
        
        # Warmup runs
        self.logger.info(f"Running {self.config.warmup_runs} warmup runs...")
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)
        
        # Actual benchmark runs
        self.logger.info(f"Running {self.config.num_runs} benchmark runs...")
        inference_times = []
        
        for i in range(self.config.num_runs):
            if self.config.precision == "fp16":
                with torch.cuda.amp.autocast():
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        if isinstance(inputs, tuple):
                            _ = model(*inputs)
                        else:
                            _ = model(inputs)
                    end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                with torch.no_grad():
                    if isinstance(inputs, tuple):
                        _ = model(*inputs)
                    else:
                        _ = model(inputs)
                end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
        
        # Calculate metrics
        avg_time = statistics.mean(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        min_time = min(inference_times)
        max_time = max(inference_times)
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        result = BenchmarkResult(
            benchmark_type=self.config.benchmark_type,
            timestamp=time.time(),
            avg_inference_time_ms=avg_time,
            std_inference_time_ms=std_time,
            min_inference_time_ms=min_time,
            max_inference_time_ms=max_time,
            avg_fps=avg_fps,
            input_shapes=self.config.input_shapes,
            num_runs=self.config.num_runs
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, result: BenchmarkResult, filename: str = None):
        """Save benchmark results."""
        if not self.config.save_results:
            return
        
        if filename is None:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.benchmark_type.value}_benchmark_{timestamp_str}.json"
        
        # Create results directory if it doesn't exist
        results_dir = Path(self.config.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = results_dir / filename
        
        # Convert result to dictionary
        result_dict = {
            "benchmark_type": result.benchmark_type.value,
            "timestamp": result.timestamp,
            "avg_inference_time_ms": result.avg_inference_time_ms,
            "std_inference_time_ms": result.std_inference_time_ms,
            "min_inference_time_ms": result.min_inference_time_ms,
            "max_inference_time_ms": result.max_inference_time_ms,
            "avg_fps": result.avg_fps,
            "memory_usage_mb": result.memory_usage_mb,
            "power_draw_w": result.power_draw_w,
            "throughput_fps": result.throughput_fps,
            "input_shapes": result.input_shapes,
            "num_runs": result.num_runs,
            "model_size_mb": result.model_size_mb
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {result_path}")
    
    def print_results(self, result: BenchmarkResult):
        """Print benchmark results."""
        if not self.config.verbose:
            return
        
        print(f"\n{'='*50}")
        print(f"Performance Benchmark Results - {result.benchmark_type.value.upper()}")
        print(f"{'='*50}")
        print(f"Inference Time: {result.avg_inference_time_ms:.2f}±{result.std_inference_time_ms:.2f} ms")
        print(f"Min Time: {result.min_inference_time_ms:.2f} ms")
        print(f"Max Time: {result.max_inference_time_ms:.2f} ms")
        print(f"FPS: {result.avg_fps:.2f}")
        
        if result.memory_usage_mb is not None:
            print(f"Memory Usage: {result.memory_usage_mb:.2f} MB")
        
        if result.power_draw_w is not None:
            print(f"Power Draw: {result.power_draw_w:.2f} W")
        
        if result.throughput_fps is not None:
            print(f"Throughput: {result.throughput_fps:.2f} FPS")


class InferenceBenchmark(PerformanceBenchmark):
    """Benchmark for model inference performance."""
    
    def __init__(self, config: BenchmarkConfig):
        if config.benchmark_type != BenchmarkType.INFERENCE:
            raise ValueError("Config must be for inference benchmark")
        super().__init__(config)
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run inference benchmark."""
        self.logger.info("Starting inference benchmark...")
        
        # Load model
        model = self._load_model()
        
        # Generate example inputs
        example_inputs = self._generate_example_inputs()
        
        # Run inference benchmark
        result = self._run_inference_benchmark(model, example_inputs)
        
        # Add model size to result if possible
        try:
            model_size = self._get_model_size(self.config.model_path)
            result.model_size_mb = model_size
        except:
            pass
        
        # Save results
        self.save_results(result)
        self.print_results(result)
        
        return result
    
    def _load_model(self):
        """Load model from path."""
        # For this example, we'll create a mock model
        # In a real implementation, this would load the actual model
        if self.config.model_path.endswith('.pt') or self.config.model_path.endswith('.pth'):
            # Load PyTorch model
            model = torch.jit.load(self.config.model_path)
        else:
            # Create a mock model for demonstration
            input_channels = self.config.input_shapes[0][1]  # Assumed to be (batch, channels, height, width)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 7 * 7, 1000)
            )
        
        return model
    
    def _generate_example_inputs(self):
        """Generate example inputs based on input shapes."""
        if len(self.config.input_shapes) == 1:
            shape = self.config.input_shapes[0]
            return torch.randn(shape).to(self.config.device)
        else:
            # Multiple input tensors
            inputs = []
            for shape in self.config.input_shapes:
                inputs.append(torch.randn(shape).to(self.config.device))
            return tuple(inputs)
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB."""
        import os
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)  # Convert to MB


class ThroughputBenchmark(PerformanceBenchmark):
    """Benchmark for system throughput performance."""
    
    def __init__(self, config: BenchmarkConfig):
        if config.benchmark_type != BenchmarkType.THROUGHPUT:
            raise ValueError("Config must be for throughput benchmark")
        super().__init__(config)
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run throughput benchmark."""
        self.logger.info("Starting throughput benchmark...")
        
        # Load model
        model = self._load_model()
        model.to(self.config.device)
        model.eval()
        
        # Generate batch inputs
        batch_inputs = self._generate_batch_inputs()
        
        # Warmup
        self.logger.info(f"Running {self.config.warmup_runs} warmup batches...")
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(batch_inputs)
        
        # Run benchmark
        self.logger.info(f"Running {self.config.num_runs} throughput runs...")
        start_time = time.time()
        
        for _ in range(self.config.num_runs):
            with torch.no_grad():
                if self.config.precision == "fp16":
                    with torch.cuda.amp.autocast():
                        _ = model(batch_inputs)
                else:
                    _ = model(batch_inputs)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_frames = self.config.num_runs * self.config.batch_size
        avg_fps = total_frames / total_time
        
        result = BenchmarkResult(
            benchmark_type=self.config.benchmark_type,
            timestamp=time.time(),
            avg_inference_time_ms=(total_time / self.config.num_runs) * 1000,
            std_inference_time_ms=0,  # Not calculated for throughput
            min_inference_time_ms=0,  # Not calculated for throughput
            max_inference_time_ms=0,  # Not calculated for throughput
            avg_fps=avg_fps,
            throughput_fps=avg_fps,
            input_shapes=self.config.input_shapes,
            num_runs=self.config.num_runs
        )
        
        self.results.append(result)
        
        # Save and print results
        self.save_results(result)
        self.print_results(result)
        
        return result
    
    def _load_model(self):
        """Load model from path."""
        # Create a mock model for demonstration
        input_channels = self.config.input_shapes[0][1]
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 1000)
        )
    
    def _generate_batch_inputs(self):
        """Generate batched inputs."""
        shape = list(self.config.input_shapes[0])
        shape[0] = self.config.batch_size  # Set batch dimension
        return torch.randn(shape).to(self.config.device)


class MemoryBenchmark(PerformanceBenchmark):
    """Benchmark for memory usage."""
    
    def __init__(self, config: BenchmarkConfig):
        if config.benchmark_type != BenchmarkType.MEMORY:
            raise ValueError("Config must be for memory benchmark")
        super().__init__(config)
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run memory benchmark."""
        import psutil
        import gc
        
        self.logger.info("Starting memory benchmark...")
        
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Load model
        model = self._load_model()
        model.to(self.config.device)
        
        # Record memory after model loading
        after_load_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run inference to trigger memory allocation
        example_inputs = self._generate_example_inputs()
        with torch.no_grad():
            _ = model(example_inputs)
        
        # Record memory after inference
        after_inference_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory usage
        model_memory = after_load_memory - initial_memory
        peak_memory = after_inference_memory - initial_memory
        
        # Clean up
        del model, example_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Record final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used_by_model = max(0, peak_memory - (final_memory - initial_memory))
        
        result = BenchmarkResult(
            benchmark_type=self.config.benchmark_type,
            timestamp=time.time(),
            avg_inference_time_ms=0,  # Not applicable for memory benchmark
            std_inference_time_ms=0,
            min_inference_time_ms=0,
            max_inference_time_ms=0,
            avg_fps=0,
            memory_usage_mb=memory_used_by_model,
            input_shapes=self.config.input_shapes,
            num_runs=self.config.num_runs
        )
        
        self.results.append(result)
        
        # Save and print results
        self.save_results(result)
        self.print_results(result)
        
        return result
    
    def _load_model(self):
        """Load model from path."""
        input_channels = self.config.input_shapes[0][1]
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 1000)
        )
    
    def _generate_example_inputs(self):
        """Generate example inputs based on input shapes."""
        shape = self.config.input_shapes[0]
        return torch.randn(shape).to(self.config.device)


class BenchmarkRunner:
    """Runner for multiple benchmarks."""
    
    def __init__(self, configs: List[BenchmarkConfig]):
        self.configs = configs
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("BenchmarkRunner")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []
        
        for i, config in enumerate(self.configs):
            self.logger.info(f"Running benchmark {i+1}/{len(self.configs)}: {config.benchmark_type.value}")
            
            if config.benchmark_type == BenchmarkType.INFERENCE:
                benchmark = InferenceBenchmark(config)
            elif config.benchmark_type == BenchmarkType.THROUGHPUT:
                benchmark = ThroughputBenchmark(config)
            elif config.benchmark_type == BenchmarkType.MEMORY:
                benchmark = MemoryBenchmark(config)
            else:
                self.logger.warning(f"Unsupported benchmark type: {config.benchmark_type.value}")
                continue
            
            result = benchmark.run_benchmark()
            results.append(result)
        
        return results
    
    def create_summary_report(self, results: List[BenchmarkResult], filename: str = "benchmark_summary.json"):
        """Create a summary report of all benchmarks."""
        summary = {
            "timestamp": time.time(),
            "benchmark_count": len(results),
            "results": []
        }
        
        for result in results:
            result_summary = {
                "benchmark_type": result.benchmark_type.value,
                "avg_inference_time_ms": result.avg_inference_time_ms,
                "std_inference_time_ms": result.std_inference_time_ms,
                "avg_fps": result.avg_fps,
                "memory_usage_mb": result.memory_usage_mb,
                "power_draw_w": result.power_draw_w,
                "throughput_fps": result.throughput_fps,
                "model_size_mb": result.model_size_mb
            }
            summary["results"].append(result_summary)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Benchmark summary report saved to {filename}")
        return summary


def create_comprehensive_benchmark_suite(model_path: str, input_shapes: List[Tuple[int, ...]]) -> BenchmarkRunner:
    """Create a comprehensive benchmark suite."""
    configs = [
        BenchmarkConfig(
            benchmark_type=BenchmarkType.INFERENCE,
            model_path=model_path,
            input_shapes=input_shapes,
            num_runs=100,
            warmup_runs=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16",
            save_results=True,
            results_path="benchmark_results/"
        ),
        BenchmarkConfig(
            benchmark_type=BenchmarkType.THROUGHPUT,
            model_path=model_path,
            input_shapes=input_shapes,
            num_runs=50,
            warmup_runs=5,
            batch_size=8,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16",
            save_results=True,
            results_path="benchmark_results/"
        ),
        BenchmarkConfig(
            benchmark_type=BenchmarkType.MEMORY,
            model_path=model_path,
            input_shapes=input_shapes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_results=True,
            results_path="benchmark_results/"
        )
    ]
    
    return BenchmarkRunner(configs)


def compare_models(model_paths: List[str], input_shapes: List[Tuple[int, ...]], model_names: List[str] = None) -> Dict[str, List[BenchmarkResult]]:
    """Compare performance across multiple models."""
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(model_paths))]
    
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nBenchmarking {model_name}: {model_path}")
        print("="*60)
        
        # Create a simple benchmark config
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.INFERENCE,
            model_path=model_path,
            input_shapes=input_shapes,
            num_runs=50,
            warmup_runs=5,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16"
        )
        
        # Run inference benchmark
        benchmark = InferenceBenchmark(config)
        result = benchmark.run_benchmark()
        
        results[model_name] = [result]
    
    return results


def example_usage():
    """Example of how to use the performance benchmarking tools."""
    print("Performance Benchmarking Tools Example")
    print("=" * 50)
    
    # Example input shapes for a model
    input_shapes = [(1, 3, 224, 224)]  # Batch size 1, 3 channels, 224x224
    
    # Create and run a comprehensive benchmark suite
    benchmark_runner = create_comprehensive_benchmark_suite(
        model_path="example_model.pt",
        input_shapes=input_shapes
    )
    
    # Run all benchmarks
    results = benchmark_runner.run_all_benchmarks()
    
    # Create a summary report
    summary = benchmark_runner.create_summary_report(results)
    print(f"\nBenchmark completed! Summary:")
    print(json.dumps(summary, indent=2))
    
    # Example of comparing models
    # model_paths = ["model_a.pt", "model_b.pt", "model_c.pt"]
    # model_names = ["EfficientNet", "ResNet", "MobileNet"]
    # comparison_results = compare_models(model_paths, input_shapes, model_names)
    # 
    # print("\nModel Comparison Results:")
    # for model_name, model_results in comparison_results.items():
    #     if model_results:
    #         result = model_results[0]
    #         print(f"{model_name}: {result.avg_fps:.2f} FPS, {result.avg_inference_time_ms:.2f} ms")


if __name__ == "__main__":
    example_usage()