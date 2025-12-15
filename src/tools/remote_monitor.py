"""
Remote hardware monitoring tools for the Physical AI & Humanoid Robotics Course

This module provides functionality to remotely monitor hardware resources,
including status, performance metrics, and health indicators.
"""
import json
import time
import threading
import requests
import psutil
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HardwareStatus:
    """Data class to represent hardware status"""
    id: str
    name: str
    type: str
    status: str  # 'online', 'offline', 'busy', 'maintenance', 'error'
    last_seen: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    temperature: Optional[float] = None
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    network_status: Optional[Dict[str, Any]] = None
    custom_metrics: Optional[Dict[str, Any]] = None


class RemoteHardwareMonitor:
    """
    Main class for remote hardware monitoring.
    
    Provides methods to monitor various hardware resources remotely,
    collect metrics, and report status.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the remote hardware monitor
        
        Args:
            base_url: Base URL for the hardware monitoring API
        """
        self.base_url = base_url
        self.monitored_hardware: Dict[str, HardwareStatus] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.status_callbacks: List[Callable[[HardwareStatus], None]] = []
        self.refresh_interval = 5  # seconds
        self.session = requests.Session()
    
    def add_hardware_resource(self, hardware_id: str, hardware_name: str, hardware_type: str) -> bool:
        """
        Add a hardware resource to be monitored
        
        Args:
            hardware_id: Unique identifier for the hardware
            hardware_name: Human-readable name of the hardware
            hardware_type: Type of hardware (e.g., 'jetson_orin', 'workstation', 'robot')
            
        Returns:
            True if successfully added, False otherwise
        """
        if hardware_id in self.monitored_hardware:
            return False
        
        initial_status = HardwareStatus(
            id=hardware_id,
            name=hardware_name,
            type=hardware_type,
            status='offline',
            last_seen=datetime.min,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0
        )
        
        self.monitored_hardware[hardware_id] = initial_status
        logger.info(f"Added hardware {hardware_id} ({hardware_name}) to monitoring")
        return True
    
    def remove_hardware_resource(self, hardware_id: str) -> bool:
        """
        Remove a hardware resource from monitoring
        
        Args:
            hardware_id: ID of the hardware to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        if hardware_id not in self.monitored_hardware:
            return False
        
        del self.monitored_hardware[hardware_id]
        logger.info(f"Removed hardware {hardware_id} from monitoring")
        return True
    
    def get_hardware_status(self, hardware_id: str) -> Optional[HardwareStatus]:
        """
        Get the current status of a specific hardware resource
        
        Args:
            hardware_id: ID of the hardware to query
            
        Returns:
            HardwareStatus object if found, None otherwise
        """
        return self.monitored_hardware.get(hardware_id)
    
    def get_all_statuses(self) -> List[HardwareStatus]:
        """
        Get the status of all monitored hardware resources
        
        Returns:
            List of HardwareStatus objects
        """
        return list(self.monitored_hardware.values())
    
    def start_monitoring(self) -> bool:
        """
        Start the monitoring process
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.monitoring_active:
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Remote hardware monitoring started")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop the monitoring process
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self.monitoring_active:
            return False
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Remote hardware monitoring stopped")
        return True
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread"""
        while self.monitoring_active:
            try:
                # Update status for each monitored hardware
                for hardware_id in list(self.monitored_hardware.keys()):
                    try:
                        new_status = self._fetch_hardware_status(hardware_id)
                        if new_status:
                            old_status = self.monitored_hardware[hardware_id]
                            self.monitored_hardware[hardware_id] = new_status
                            
                            # Call any registered callbacks if status changed significantly
                            if (abs(new_status.cpu_usage - old_status.cpu_usage) > 10.0 or
                                abs(new_status.memory_usage - old_status.memory_usage) > 10.0 or
                                new_status.status != old_status.status):
                                self._trigger_status_callbacks(new_status)
                    except Exception as e:
                        logger.error(f"Error updating status for {hardware_id}: {e}")
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for the refresh interval
            time.sleep(self.refresh_interval)
    
    def _fetch_hardware_status(self, hardware_id: str) -> Optional[HardwareStatus]:
        """
        Fetch hardware status from remote API
        
        Args:
            hardware_id: ID of the hardware to fetch status for
            
        Returns:
            Updated HardwareStatus object, or None if failed
        """
        # In a real implementation, this would make an API call to the hardware
        # For simulation, we'll create mock data
        try:
            # Placeholder: Make an API call to get hardware status
            # response = self.session.get(f"{self.base_url}/hardware/{hardware_id}/status")
            
            # For now, return mock data
            return self._create_mock_status(hardware_id)
        except Exception as e:
            logger.error(f"Failed to fetch status for {hardware_id}: {e}")
            # Update status to offline if we can't reach the hardware
            current_status = self.monitored_hardware.get(hardware_id)
            if current_status:
                return HardwareStatus(
                    id=current_status.id,
                    name=current_status.name,
                    type=current_status.type,
                    status='offline',
                    last_seen=datetime.now(),
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    disk_usage=0.0,
                    custom_metrics={'error': str(e)}
                )
            return None
    
    def _create_mock_status(self, hardware_id: str) -> HardwareStatus:
        """
        Create mock status data for testing purposes
        
        Args:
            hardware_id: ID of the hardware to create mock status for
            
        Returns:
            HardwareStatus object with mock data
        """
        # Simulate realistic hardware metrics
        import random
        
        cpu_usage = random.uniform(5.0, 85.0)
        memory_usage = random.uniform(10.0, 90.0)
        disk_usage = random.uniform(20.0, 70.0)
        temperature = random.uniform(30.0, 70.0) if random.random() > 0.7 else random.uniform(70.0, 85.0)
        gpu_usage = random.uniform(10.0, 95.0) if random.random() > 0.3 else 0.0
        gpu_memory = random.uniform(20.0, 90.0) if gpu_usage > 0 else 0.0
        
        # Determine status based on metrics
        status = 'online'
        if temperature > 80:
            status = 'warning'
        elif temperature > 85:
            status = 'error'
        elif cpu_usage > 90 or memory_usage > 95:
            status = 'busy'
        
        return HardwareStatus(
            id=hardware_id,
            name=f"Mock Hardware {hardware_id}",
            type="mock_device",
            status=status,
            last_seen=datetime.now(),
            cpu_usage=round(cpu_usage, 2),
            memory_usage=round(memory_usage, 2),
            disk_usage=round(disk_usage, 2),
            temperature=round(temperature, 2),
            gpu_usage=round(gpu_usage, 2) if gpu_usage > 0 else None,
            gpu_memory=round(gpu_memory, 2) if gpu_memory > 0 else None
        )
    
    def register_status_callback(self, callback: Callable[[HardwareStatus], None]) -> None:
        """
        Register a callback function to be called when hardware status changes
        
        Args:
            callback: Function to call with HardwareStatus when changes occur
        """
        self.status_callbacks.append(callback)
    
    def _trigger_status_callbacks(self, status: HardwareStatus) -> None:
        """
        Trigger all registered status callbacks
        
        Args:
            status: Updated HardwareStatus to pass to callbacks
        """
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def get_system_performance_metrics(self, hardware_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance metrics for a specific hardware resource
        
        Args:
            hardware_id: ID of the hardware to get metrics for
            
        Returns:
            Dictionary with performance metrics, or None if unavailable
        """
        # In a real implementation, this would call a specific API endpoint
        # For simulation, return mock metrics
        status = self.monitored_hardware.get(hardware_id)
        if not status:
            return None
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage_percent': status.cpu_usage,
                'count': 8,  # Mock value
                'frequency': 2.2  # GHz, mock value
            },
            'memory': {
                'usage_percent': status.memory_usage,
                'total_gb': 32.0,  # Mock value
                'available_gb': 32.0 * (1 - status.memory_usage/100)
            },
            'disk': {
                'usage_percent': status.disk_usage,
                'total_gb': 512.0,  # Mock value
                'available_gb': 512.0 * (1 - status.disk_usage/100)
            },
            'temperature': {
                'current_celsius': status.temperature,
                'critical_threshold': 85.0
            },
            'network': {
                'bandwidth_mbps': 1000.0,  # Mock value
                'latency_ms': 5.0  # Mock value
            }
        }
    
    def trigger_hardware_action(self, hardware_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger an action on a specific hardware resource
        
        Args:
            hardware_id: ID of the hardware to perform action on
            action: Action to perform (e.g., 'restart', 'calibrate', 'update')
            params: Optional parameters for the action
            
        Returns:
            True if action was successfully triggered, False otherwise
        """
        # In a real implementation, this would send a command to the hardware
        logger.info(f"Triggering action '{action}' on hardware {hardware_id} with params: {params}")
        
        # For simulation purposes, just log the action
        # In a real implementation, you would make an API call to perform the action
        return True


class HardwareHealthDashboard:
    """
    A simple dashboard class for displaying hardware health information.
    """
    
    def __init__(self, monitor: RemoteHardwareMonitor):
        """
        Initialize the hardware health dashboard
        
        Args:
            monitor: RemoteHardwareMonitor instance to get data from
        """
        self.monitor = monitor
    
    def generate_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report of all monitored hardware
        
        Returns:
            Dictionary containing the status report
        """
        all_statuses = self.monitor.get_all_statuses()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_hardware_count': len(all_statuses),
            'online_count': len([s for s in all_statuses if s.status == 'online']),
            'offline_count': len([s for s in all_statuses if s.status == 'offline']),
            'busy_count': len([s for s in all_statuses if s.status == 'busy']),
            'warning_count': len([s for s in all_statuses if s.status == 'warning']),
            'error_count': len([s for s in all_statuses if s.status == 'error']),
            'hardware_details': []
        }
        
        for status in all_statuses:
            detail = {
                'id': status.id,
                'name': status.name,
                'type': status.type,
                'status': status.status,
                'last_seen': status.last_seen.isoformat(),
                'metrics': {
                    'cpu_usage': status.cpu_usage,
                    'memory_usage': status.memory_usage,
                    'disk_usage': status.disk_usage,
                    'temperature': status.temperature
                }
            }
            if status.gpu_usage is not None:
                detail['metrics']['gpu_usage'] = status.gpu_usage
            if status.gpu_memory is not None:
                detail['metrics']['gpu_memory'] = status.gpu_memory
            
            report['hardware_details'].append(detail)
        
        return report
    
    def print_status_summary(self) -> None:
        """Print a text summary of the hardware status"""
        report = self.generate_status_report()
        
        print("\n" + "="*60)
        print("REMOTE HARDWARE MONITORING DASHBOARD")
        print("="*60)
        print(f"Report Time: {report['timestamp']}")
        print(f"Total Hardware Resources: {report['total_hardware_count']}")
        print(f"  - Online: {report['online_count']}")
        print(f"  - Busy: {report['busy_count']}")
        print(f"  - Warning: {report['warning_count']}")
        print(f"  - Error: {report['error_count']}")
        print(f"  - Offline: {report['offline_count']}")
        print("\nDetailed Status:")
        
        for detail in report['hardware_details']:
            status_symbol = {
                'online': '✅',
                'busy': '⏳',
                'warning': '⚠️',
                'error': '❌',
                'offline': '🔴'
            }.get(detail['status'], '?')
            
            print(f"  {status_symbol} {detail['name']} ({detail['type']}) - {detail['status']}")
            print(f"    CPU: {detail['metrics']['cpu_usage']:.1f}% | "
                  f"Mem: {detail['metrics']['memory_usage']:.1f}% | "
                  f"Disk: {detail['metrics']['disk_usage']:.1f}%")
            if 'temperature' in detail['metrics'] and detail['metrics']['temperature'] is not None:
                print(f"    Temp: {detail['metrics']['temperature']:.1f}°C")
        
        print("="*60)


def setup_remote_monitoring(base_url: str = "http://localhost:8080") -> RemoteHardwareMonitor:
    """
    Set up remote monitoring with default configurations for the course.
    
    Args:
        base_url: Base URL for the hardware monitoring API
        
    Returns:
        Configured RemoteHardwareMonitor instance
    """
    monitor = RemoteHardwareMonitor(base_url)
    
    # Add common hardware resources used in the course
    monitor.add_hardware_resource("jetson_orin_01", "Jetson Orin Development Kit #1", "jetson_orin")
    monitor.add_hardware_resource("jetson_orin_02", "Jetson Orin Development Kit #2", "jetson_orin")
    monitor.add_hardware_resource("robot_unitree_h1_01", "Unitree H1 Robot #1", "humanoid_robot")
    monitor.add_hardware_resource("simulation_workstation_01", "Simulation Workstation #1", "workstation")
    
    logger.info("Remote monitoring setup complete with default hardware resources")
    
    # Start monitoring
    monitor.start_monitoring()
    
    return monitor


# Example usage and testing
if __name__ == "__main__":
    # Set up monitoring
    monitor = setup_remote_monitoring()
    
    # Create dashboard
    dashboard = HardwareHealthDashboard(monitor)
    
    # Print initial status
    dashboard.print_status_summary()
    
    # Let it run for a few seconds to see updates
    try:
        time.sleep(10)
        dashboard.print_status_summary()
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        monitor.stop_monitoring()
        print("Monitoring stopped.")