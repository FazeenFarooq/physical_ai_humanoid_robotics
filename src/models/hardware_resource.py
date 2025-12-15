"""
HardwareResource entity model for Physical AI & Humanoid Robotics Course

This module defines the HardwareResource entity that represents
physical hardware resources in the course infrastructure.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from datetime import datetime


class HardwareType(Enum):
    """Enumeration of hardware resource types"""
    WORKSTATION = "workstation"
    JETSON_ORIN = "jetson_orin"
    ROBOT_PLATFORM = "robot_platform"
    SIMULATION_WORKSTATION = "simulation_workstation"
    SENSORS = "sensors"
    OTHER = "other"


class HardwareStatus(Enum):
    """Enumeration of hardware resource statuses"""
    AVAILABLE = "available"
    RESERVED = "reserved"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    FAULTY = "faulty"


@dataclass
class HardwareResource:
    """
    Represents a hardware resource in the Physical AI & Humanoid Robotics course.
    
    Attributes:
        id (str): Unique identifier for the hardware resource
        type (HardwareType): Type of hardware (workstation, jetson_orin, etc.)
        model (str): Specific model and specifications
        location (str): Physical location of the hardware
        status (HardwareStatus): Current status (available, reserved, in_use, etc.)
        reservation (Optional[str]): Current reservation information
        owner (Optional[str]): Person responsible for maintenance
        last_calibration (Optional[datetime]): Date of last calibration
        availability_schedule (Optional[List[dict]]): When the hardware is available
        specifications (Optional[dict]): Detailed hardware specifications
        notes (Optional[str]): Additional notes about the hardware
    """
    
    id: str
    type: HardwareType
    model: str
    location: str
    status: HardwareStatus
    reservation: Optional[str] = None
    owner: Optional[str] = None
    last_calibration: Optional[datetime] = None
    availability_schedule: Optional[List[dict]] = None
    specifications: Optional[dict] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate the HardwareResource after initialization"""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("HardwareResource ID must be a non-empty string")
        
        if not self.model or not isinstance(self.model, str):
            raise ValueError("HardwareResource model must be a non-empty string")
        
        if not self.location or not isinstance(self.location, str):
            raise ValueError("HardwareResource location must be a non-empty string")
    
    def is_available(self) -> bool:
        """Check if the hardware is available for use"""
        return self.status == HardwareStatus.AVAILABLE
    
    def reserve(self, reservation_id: str) -> bool:
        """Reserve this hardware resource if available"""
        if self.status == HardwareStatus.AVAILABLE:
            self.status = HardwareStatus.RESERVED
            self.reservation = reservation_id
            return True
        return False
    
    def release_reservation(self) -> bool:
        """Release the current reservation if it exists"""
        if self.status == HardwareStatus.RESERVED:
            self.status = HardwareStatus.AVAILABLE
            self.reservation = None
            return True
        return False
    
    def mark_in_use(self) -> bool:
        """Mark the hardware as currently in use"""
        if self.status in [HardwareStatus.AVAILABLE, HardwareStatus.RESERVED]:
            self.status = HardwareStatus.IN_USE
            return True
        return False
    
    def mark_available(self) -> bool:
        """Mark the hardware as available after use"""
        self.status = HardwareStatus.AVAILABLE
        self.reservation = None
        return True
    
    def mark_maintenance(self) -> bool:
        """Mark the hardware as needing maintenance"""
        self.status = HardwareStatus.MAINTENANCE
        self.reservation = None
        return True
    
    def mark_faulty(self) -> bool:
        """Mark the hardware as faulty"""
        self.status = HardwareStatus.FAULTY
        self.reservation = None
        return True