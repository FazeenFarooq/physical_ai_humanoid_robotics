"""
Hardware resource management system for Jetson Orin and robot platforms
in the Physical AI & Humanoid Robotics course.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .entities import HardwareResource, HardwareStatus
from .student_management_service import StudentManagementService
import uuid


class HardwareResourceManager:
    """Management system for hardware resources in the Physical AI course"""
    
    def __init__(self, student_service: StudentManagementService):
        self.student_service = student_service
        self.hardware_resources: Dict[str, HardwareResource] = {}
        self.reservation_history: List[Dict] = []  # Track reservation history for analytics
        
    def add_jetson_orin_kit(self, model: str, location: str, 
                           serial_number: Optional[str] = None) -> HardwareResource:
        """Add a Jetson Orin kit to the system"""
        resource_id = f"jetson_orin_{serial_number}" if serial_number else f"jetson_orin_{str(uuid.uuid4())[:8]}"
        
        hardware = HardwareResource(
            id=resource_id,
            type="Jetson Orin",
            model=model,
            location=location
        )
        
        self.hardware_resources[resource_id] = hardware
        return hardware
    
    def add_robot_platform(self, robot_type: str, model: str, location: str,
                          serial_number: Optional[str] = None) -> HardwareResource:
        """Add a robot platform to the system"""
        resource_id = f"{robot_type.lower()}_{serial_number}" if serial_number else f"{robot_type.lower()}_{str(uuid.uuid4())[:8]}"
        
        hardware = HardwareResource(
            id=resource_id,
            type=robot_type,
            model=model,
            location=location
        )
        
        self.hardware_resources[resource_id] = hardware
        return hardware
    
    def reserve_hardware(self, resource_id: str, student_id: str, duration_hours: int = 2) -> bool:
        """Reserve a hardware resource for a student for a specific duration"""
        resource = self.hardware_resources.get(resource_id)
        if resource and resource.status == HardwareStatus.AVAILABLE:
            resource.status = HardwareStatus.RESERVED
            resource.reservation = f"{student_id}_{datetime.now().isoformat()}"
            return True
        return False
    
    def start_hardware_session(self, resource_id: str, student_id: str) -> bool:
        """Start a hardware session when a student begins using the hardware"""
        resource = self.hardware_resources.get(resource_id)
        if resource and (resource.status == HardwareStatus.RESERVED or resource.status == HardwareStatus.AVAILABLE):
            resource.status = HardwareStatus.IN_USE
            # Log the session start
            self.reservation_history.append({
                'resource_id': resource_id,
                'student_id': student_id,
                'start_time': datetime.now(),
                'status': 'started'
            })
            return True
        return False
    
    def release_hardware(self, resource_id: str, student_id: str) -> bool:
        """Release hardware back to available state after student use"""
        resource = self.hardware_resources.get(resource_id)
        if resource and resource.status == HardwareStatus.IN_USE:
            resource.status = HardwareStatus.AVAILABLE
            resource.reservation = None
            
            # Log the session end
            self.reservation_history.append({
                'resource_id': resource_id,
                'student_id': student_id,
                'end_time': datetime.now(),
                'status': 'released'
            })
            return True
        return False
    
    def schedule_maintenance(self, resource_id: str, scheduled_date: datetime) -> bool:
        """Schedule maintenance for a hardware resource"""
        resource = self.hardware_resources.get(resource_id)
        if resource:
            # If currently in use, mark for maintenance after release
            if resource.status == HardwareStatus.IN_USE:
                resource.status = HardwareStatus.MAINTENANCE  # Will be set after release
            else:
                resource.status = HardwareStatus.MAINTENANCE
            resource.last_calibration = scheduled_date
            return True
        return False
    
    def mark_faulty(self, resource_id: str, reason: str) -> bool:
        """Mark a hardware resource as faulty"""
        resource = self.hardware_resources.get(resource_id)
        if resource:
            resource.status = HardwareStatus.FAULTY
            # Log the fault
            self.reservation_history.append({
                'resource_id': resource_id,
                'reason': reason,
                'timestamp': datetime.now(),
                'status': 'faulty'
            })
            return True
        return False
    
    def get_available_jetscan_orin(self) -> List[HardwareResource]:
        """Get all available Jetson Orin hardware"""
        return [h for h in self.hardware_resources.values() 
                if h.type == "Jetson Orin" and h.status == HardwareStatus.AVAILABLE]
    
    def get_available_robot_platforms(self) -> List[HardwareResource]:
        """Get all available robot platforms"""
        return [h for h in self.hardware_resources.values() 
                if "robot" in h.type.lower() and h.status == HardwareStatus.AVAILABLE]
    
    def get_hardware_status_report(self) -> Dict:
        """Generate a status report of all hardware resources"""
        report = {
            'total_resources': len(self.hardware_resources),
            'by_status': {
                'available': 0,
                'reserved': 0,
                'in_use': 0,
                'maintenance': 0,
                'faulty': 0
            },
            'by_type': {}
        }
        
        for resource in self.hardware_resources.values():
            # Count by status
            report['by_status'][resource.status.value] += 1
            
            # Count by type
            if resource.type not in report['by_type']:
                report['by_type'][resource.type] = 0
            report['by_type'][resource.type] += 1
        
        return report
    
    def get_reservation_history(self, student_id: Optional[str] = None) -> List[Dict]:
        """Get reservation history, optionally filtered by student"""
        if student_id:
            return [r for r in self.reservation_history if r.get('student_id') == student_id]
        return self.reservation_history