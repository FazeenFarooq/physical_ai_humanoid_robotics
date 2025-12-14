"""
Computational Resource Monitoring for the Physical AI & Humanoid Robotics Course.

This module provides tools for monitoring computational resources on the Jetson Orin
and other platforms, enabling students to understand and optimize their AI applications
for embedded robotics systems with resource constraints.
"""

import psutil
import GPUtil
import time
import threading
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import os
import subprocess
import logging


@dataclass
class ResourceMetrics:
    """Data class for resource monitoring metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    disk_percent: float = 0.0
    temperature: Optional[float] = None
    power_draw: Optional[float] = None  # Not available on all platforms


class ResourceMonitor:
    """Monitor for system computational resources."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history: List[ResourceMetrics] = []
        self.callbacks: List[Callable[[ResourceMetrics], None]] = []
        self.logger = self._setup_logger()
        self._lock = threading.Lock()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the resource monitor."""
        logger = logging.getLogger("ResourceMonitor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        if self.is_monitoring:
            self.logger.warning("Resource monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"Error in metrics callback: {e}")
                
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk_percent = psutil.disk_usage('/').percent
        
        # GPU metrics (if available)
        gpu_percent = None
        gpu_memory_percent = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assume single GPU
            gpu_percent = gpu.load * 100
            gpu_memory_percent = gpu.memoryUtil * 100
            gpu_memory_used_gb = gpu.memoryUsed / 1024
            gpu_memory_total_gb = gpu.memoryTotal / 1024
        
        # Temperature (if available)
        temperature = self._get_temperature()
        
        # Power draw (Jetson-specific)
        power_draw = self._get_jetson_power_draw()
        
        return ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            disk_percent=disk_percent,
            temperature=temperature,
            power_draw=power_draw
        )
    
    def _get_temperature(self) -> Optional[float]:
        """Get system temperature if available."""
        try:
            # Try to get temperature from different sources
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
        except:
            pass
        
        # For other systems, try sensors command if available
        try:
            result = subprocess.run(['sensors'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse temperature from sensors output
                for line in result.stdout.split('\n'):
                    if 'temp1:' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            temp_str = parts[1].replace('+', '').replace('°C', '')
                            try:
                                return float(temp_str)
                            except ValueError:
                                continue
        except:
            pass
        
        return None
    
    def _get_jetson_power_draw(self) -> Optional[float]:
        """Get power draw on Jetson platforms."""
        # This is platform-specific for Jetson
        try:
            # Try to read from Jetson power monitor
            # Path may vary depending on Jetson model
            if os.path.exists('/sys/bus/i2c/drivers/ina3221/0-0040/iio:device0/in_power0_input'):
                with open('/sys/bus/i2c/drivers/ina3221/0-0040/iio:device0/in_power0_input', 'r') as f:
                    power_mw = int(f.read().strip())
                    return power_mw / 1000.0  # Convert mW to W
        except:
            pass
        
        return None
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get the most recent metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            else:
                return self._collect_metrics()
    
    def get_average_metrics(self) -> Optional[ResourceMetrics]:
        """Get average metrics over the entire monitoring period."""
        with self._lock:
            if not self.metrics_history:
                return None
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
            
            # For GPU metrics, filter out None values
            gpu_percents = [m.gpu_percent for m in self.metrics_history if m.gpu_percent is not None]
            avg_gpu = sum(gpu_percents) / len(gpu_percents) if gpu_percents else None
            
            gpu_memory_percents = [m.gpu_memory_percent for m in self.metrics_history if m.gpu_memory_percent is not None]
            avg_gpu_memory = sum(gpu_memory_percents) / len(gpu_memory_percents) if gpu_memory_percents else None
            
            # Use the most recent values for static metrics
            latest = self.metrics_history[-1]
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                memory_used_gb=latest.memory_used_gb,
                memory_total_gb=latest.memory_total_gb,
                gpu_percent=avg_gpu,
                gpu_memory_percent=avg_gpu_memory,
                gpu_memory_used_gb=latest.gpu_memory_used_gb,
                gpu_memory_total_gb=latest.gpu_memory_total_gb,
                disk_percent=latest.disk_percent,
                temperature=latest.temperature,
                power_draw=latest.power_draw
            )
    
    def register_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Register a callback for metrics updates."""
        self.callbacks.append(callback)
    
    def save_history_to_csv(self, filepath: str):
        """Save metrics history to CSV file."""
        with self._lock:
            if not self.metrics_history:
                self.logger.warning("No metrics history to save")
                return
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb', 
                    'memory_total_gb', 'gpu_percent', 'gpu_memory_percent', 
                    'gpu_memory_used_gb', 'gpu_memory_total_gb', 'disk_percent', 
                    'temperature', 'power_draw'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for metric in self.metrics_history:
                    writer.writerow({
                        'timestamp': metric.timestamp,
                        'cpu_percent': metric.cpu_percent,
                        'memory_percent': metric.memory_percent,
                        'memory_used_gb': metric.memory_used_gb,
                        'memory_total_gb': metric.memory_total_gb,
                        'gpu_percent': metric.gpu_percent,
                        'gpu_memory_percent': metric.gpu_memory_percent,
                        'gpu_memory_used_gb': metric.gpu_memory_used_gb,
                        'gpu_memory_total_gb': metric.gpu_memory_total_gb,
                        'disk_percent': metric.disk_percent,
                        'temperature': metric.temperature,
                        'power_draw': metric.power_draw
                    })
            
            self.logger.info(f"Metrics history saved to {filepath}")
    
    def save_history_to_json(self, filepath: str):
        """Save metrics history to JSON file."""
        with self._lock:
            if not self.metrics_history:
                self.logger.warning("No metrics history to save")
                return
            
            serializable_history = []
            for metric in self.metrics_history:
                metric_dict = {
                    'timestamp': metric.timestamp,
                    'cpu_percent': metric.cpu_percent,
                    'memory_percent': metric.memory_percent,
                    'memory_used_gb': metric.memory_used_gb,
                    'memory_total_gb': metric.memory_total_gb,
                    'gpu_percent': metric.gpu_percent,
                    'gpu_memory_percent': metric.gpu_memory_percent,
                    'gpu_memory_used_gb': metric.gpu_memory_used_gb,
                    'gpu_memory_total_gb': metric.gpu_memory_total_gb,
                    'disk_percent': metric.disk_percent,
                    'temperature': metric.temperature,
                    'power_draw': metric.power_draw
                }
                serializable_history.append(metric_dict)
            
            with open(filepath, 'w') as jsonfile:
                json.dump(serializable_history, jsonfile, indent=2)
            
            self.logger.info(f"Metrics history saved to {filepath}")
    
    def get_resource_alerts(self) -> List[Dict[str, Any]]:
        """Get any resource usage alerts."""
        alerts = []
        
        with self._lock:
            if not self.metrics_history:
                return alerts
            
            latest = self.metrics_history[-1]
            
            # Check CPU usage
            if latest.cpu_percent > 90:
                alerts.append({
                    'type': 'high_cpu',
                    'severity': 'warning',
                    'message': f'High CPU usage: {latest.cpu_percent:.1f}%',
                    'timestamp': latest.timestamp
                })
            
            # Check memory usage
            if latest.memory_percent > 85:
                alerts.append({
                    'type': 'high_memory',
                    'severity': 'warning',
                    'message': f'High memory usage: {latest.memory_percent:.1f}%',
                    'timestamp': latest.timestamp
                })
            
            # Check GPU usage (if available)
            if latest.gpu_percent and latest.gpu_percent > 95:
                alerts.append({
                    'type': 'high_gpu',
                    'severity': 'warning',
                    'message': f'High GPU usage: {latest.gpu_percent:.1f}%',
                    'timestamp': latest.timestamp
                })
            
            # Check temperature (if available)
            if latest.temperature and latest.temperature > 80:
                alerts.append({
                    'type': 'high_temperature',
                    'severity': 'critical',
                    'message': f'High temperature: {latest.temperature:.1f}°C',
                    'timestamp': latest.timestamp
                })
        
        return alerts


class ResourceMonitorGUI:
    """Simple GUI for resource monitoring (requires matplotlib)."""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.fig = None
        self.axs = None
        self.plt = None
        self.lines = {}
        self.x_data = []
        self.y_data = {}
        
    def _import_plt(self):
        """Import matplotlib if available."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            self.plt = plt
            return True
        except ImportError:
            self.logger.error("matplotlib not available. Install with: pip install matplotlib")
            return False
    
    def setup_plot(self):
        """Setup the plotting area."""
        if not self._import_plt():
            return False
        
        self.fig, self.axs = self.plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-Time Resource Monitor')
        
        # Initialize line objects
        self.lines['cpu'] = self.axs[0, 0].plot([], [], 'b-', label='CPU %')[0]
        self.axs[0, 0].set_title('CPU Usage')
        self.axs[0, 0].set_ylabel('%')
        self.axs[0, 0].set_ylim(0, 100)
        self.axs[0, 0].legend()
        
        self.lines['memory'] = self.axs[0, 1].plot([], [], 'g-', label='Memory %')[0]
        self.axs[0, 1].set_title('Memory Usage')
        self.axs[0, 1].set_ylabel('%')
        self.axs[0, 1].set_ylim(0, 100)
        self.axs[0, 1].legend()
        
        self.lines['gpu'] = self.axs[1, 0].plot([], [], 'r-', label='GPU %')[0]
        self.axs[1, 0].set_title('GPU Usage')
        self.axs[1, 0].set_ylabel('%')
        self.axs[1, 0].set_ylim(0, 100)
        self.axs[1, 0].legend()
        
        self.lines['temperature'] = self.axs[1, 1].plot([], [], 'm-', label='Temp °C')[0]
        self.axs[1, 1].set_title('Temperature')
        self.axs[1, 1].set_ylabel('°C')
        self.axs[1, 1].set_ylim(0, 100)
        self.axs[1, 1].legend()
        
        return True
    
    def update_plot(self, frame):
        """Update the plot with new data."""
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        
        # Add timestamp
        self.x_data.append(datetime.fromtimestamp(current_metrics.timestamp))
        
        # Add metric values
        if 'cpu' not in self.y_data:
            self.y_data['cpu'] = []
            self.y_data['memory'] = []
            self.y_data['gpu'] = []
            self.y_data['temperature'] = []
        
        self.y_data['cpu'].append(current_metrics.cpu_percent)
        self.y_data['memory'].append(current_metrics.memory_percent)
        self.y_data['gpu'].append(current_metrics.gpu_percent or 0)
        self.y_data['temperature'].append(current_metrics.temperature or 0)
        
        # Keep only the last 50 points
        if len(self.x_data) > 50:
            self.x_data = self.x_data[-50:]
            for key in self.y_data:
                self.y_data[key] = self.y_data[key][-50:]
        
        # Update lines
        self.lines['cpu'].set_data(self.x_data, self.y_data['cpu'])
        self.lines['memory'].set_data(self.x_data, self.y_data['memory'])
        self.lines['gpu'].set_data(self.x_data, self.y_data['gpu'])
        self.lines['temperature'].set_data(self.x_data, self.y_data['temperature'])
        
        # Adjust x-axis limits
        if len(self.x_data) > 0:
            self.axs[0, 0].set_xlim(self.x_data[0], self.x_data[-1])
            self.axs[0, 1].set_xlim(self.x_data[0], self.x_data[-1])
            self.axs[1, 0].set_xlim(self.x_data[0], self.x_data[-1])
            self.axs[1, 1].set_xlim(self.x_data[0], self.x_data[-1])
    
    def run(self, interval_ms: int = 1000):
        """Run the GUI display."""
        if not self.setup_plot():
            return
        
        # Create animation
        ani = self.plt.animation.FuncAnimation(
            self.fig, self.update_plot, interval=interval_ms, cache_frame_data=False
        )
        
        try:
            self.plt.show()
        except KeyboardInterrupt:
            print("GUI closed by user")


def create_resource_profile_report(monitor: ResourceMonitor, app_name: str) -> Dict[str, Any]:
    """Create a resource profile report for an application."""
    avg_metrics = monitor.get_average_metrics()
    
    if not avg_metrics:
        return {"error": "No metrics available"}
    
    report = {
        "app_name": app_name,
        "report_timestamp": datetime.now().isoformat(),
        "duration": len(monitor.metrics_history) * monitor.interval if monitor.metrics_history else 0,
        "average_metrics": {
            "cpu_percent": avg_metrics.cpu_percent,
            "memory_percent": avg_metrics.memory_percent,
            "gpu_percent": avg_metrics.gpu_percent,
            "gpu_memory_percent": avg_metrics.gpu_memory_percent,
            "temperature_max": max((m.temperature or 0 for m in monitor.metrics_history if m.temperature is not None), default=0),
            "power_draw_max": max((m.power_draw or 0 for m in monitor.metrics_history if m.power_draw is not None), default=0)
        },
        "peak_metrics": {
            "cpu_percent": max((m.cpu_percent for m in monitor.metrics_history), default=0),
            "gpu_percent": max((m.gpu_percent or 0 for m in monitor.metrics_history if m.gpu_percent is not None), default=0),
            "memory_percent": max((m.memory_percent for m in monitor.metrics_history), default=0),
        },
        "resource_efficiency": {}
    }
    
    # Calculate efficiency metrics
    if avg_metrics.gpu_percent and avg_metrics.gpu_percent > 0:
        report["resource_efficiency"]["compute_utilization"] = avg_metrics.gpu_percent
    else:
        report["resource_efficiency"]["compute_utilization"] = avg_metrics.cpu_percent
    
    return report


def example_usage():
    """Example of how to use the resource monitoring tools."""
    print("Starting resource monitoring example...")
    
    # Create a resource monitor
    monitor = ResourceMonitor(interval=0.5)  # Check every 0.5 seconds
    
    # Define a callback to print alerts
    def alert_callback(metrics):
        alerts = []
        if metrics.cpu_percent > 90:
            alerts.append(f"CPU: {metrics.cpu_percent:.1f}%")
        if metrics.gpu_percent and metrics.gpu_percent > 90:
            alerts.append(f"GPU: {metrics.gpu_percent:.1f}%")
        if alerts:
            print(f"ALERT: {' | '.join(alerts)}")
    
    # Register the callback
    monitor.register_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("Monitoring system resources for 10 seconds...")
    time.sleep(10)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get average metrics
    avg_metrics = monitor.get_average_metrics()
    print(f"\nAverage resource usage over 10 seconds:")
    print(f"  CPU: {avg_metrics.cpu_percent:.1f}%")
    print(f"  Memory: {avg_metrics.memory_percent:.1f}%")
    if avg_metrics.gpu_percent is not None:
        print(f"  GPU: {avg_metrics.gpu_percent:.1f}%")
    if avg_metrics.temperature is not None:
        print(f"  Temperature: {avg_metrics.temperature:.1f}°C")
    
    # Get resource alerts
    alerts = monitor.get_resource_alerts()
    if alerts:
        print(f"\nResource alerts generated: {len(alerts)}")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    
    # Save metrics to files
    monitor.save_history_to_json("resource_monitoring.json")
    monitor.save_history_to_csv("resource_monitoring.csv")
    print("\nResource history saved to 'resource_monitoring.json' and 'resource_monitoring.csv'")
    
    # Create a resource profile report
    report = create_resource_profile_report(monitor, "AI Perception Pipeline")
    print(f"\nResource profile report summary:")
    print(f"  App: {report['app_name']}")
    print(f"  Duration: {report['duration']:.1f}s")
    print(f"  Average CPU: {report['average_metrics']['cpu_percent']:.1f}%")
    print(f"  Average Memory: {report['average_metrics']['memory_percent']:.1f}%")


if __name__ == "__main__":
    example_usage()