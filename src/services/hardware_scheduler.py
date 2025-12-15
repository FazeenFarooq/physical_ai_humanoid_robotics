"""
Hardware scheduling system for the Physical AI & Humanoid Robotics Course

This module implements the hardware scheduling functionality to manage
reservations and availability of physical hardware resources.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ..models.hardware_resource import HardwareResource, HardwareStatus


class HardwareScheduler:
    """
    Manages scheduling and reservation of hardware resources for the course.
    
    The scheduler handles:
    - Hardware resource reservations
    - Availability checking
    - Conflict detection
    - Reservation management
    """
    
    def __init__(self):
        """Initialize the hardware scheduler"""
        self.hardware_resources: Dict[str, HardwareResource] = {}
        self.reservations: Dict[str, Dict] = {}  # reservation_id -> reservation details
        self.reservation_history: List[Dict] = []
    
    def add_hardware_resource(self, hardware_resource: HardwareResource) -> bool:
        """Add a new hardware resource to the scheduler"""
        if hardware_resource.id in self.hardware_resources:
            return False
        
        self.hardware_resources[hardware_resource.id] = hardware_resource
        return True
    
    def get_hardware_resource(self, resource_id: str) -> Optional[HardwareResource]:
        """Get a hardware resource by its ID"""
        return self.hardware_resources.get(resource_id)
    
    def get_available_resources(self, hardware_type: Optional[str] = None) -> List[HardwareResource]:
        """Get all available hardware resources, optionally filtered by type"""
        available_resources = [
            resource for resource in self.hardware_resources.values()
            if resource.is_available()
        ]
        
        if hardware_type:
            available_resources = [
                resource for resource in available_resources
                if resource.type.value == hardware_type
            ]
        
        return available_resources
    
    def check_availability(
        self, 
        resource_id: str, 
        start_time: datetime, 
        duration: timedelta
    ) -> Tuple[bool, str]:
        """
        Check if a hardware resource is available for a given time period
        
        Args:
            resource_id: ID of the hardware resource
            start_time: Start time of the requested reservation
            duration: Duration of the requested reservation
            
        Returns:
            Tuple of (is_available, reason) where is_available is a boolean
            and reason is a string explaining the result
        """
        if resource_id not in self.hardware_resources:
            return False, f"Hardware resource {resource_id} does not exist"
        
        resource = self.hardware_resources[resource_id]
        
        if not resource.is_available():
            return False, f"Hardware resource {resource_id} is not available (status: {resource.status.value})"
        
        # Check for any overlapping reservations
        end_time = start_time + duration
        for reservation_id, reservation in self.reservations.items():
            if reservation['resource_id'] == resource_id:
                reservation_start = reservation['start_time']
                reservation_end = reservation['start_time'] + reservation['duration']
                
                # Check for overlap
                if start_time < reservation_end and end_time > reservation_start:
                    return False, f"Hardware resource {resource_id} has conflicting reservation"
        
        return True, "Available"
    
    def make_reservation(
        self, 
        resource_id: str, 
        user_id: str, 
        start_time: datetime, 
        duration: timedelta,
        purpose: str = ""
    ) -> Tuple[bool, str]:
        """
        Make a reservation for a hardware resource
        
        Args:
            resource_id: ID of the hardware resource to reserve
            user_id: ID of the user making the reservation
            start_time: Start time of the reservation
            duration: Duration of the reservation
            purpose: Optional purpose for the reservation
            
        Returns:
            Tuple of (success, message) where success is a boolean
            and message is a string with details
        """
        is_available, reason = self.check_availability(resource_id, start_time, duration)
        
        if not is_available:
            return False, reason
        
        # Generate a unique reservation ID
        reservation_id = f"res_{resource_id}_{int(start_time.timestamp())}"
        
        # Create the reservation
        reservation = {
            'id': reservation_id,
            'resource_id': resource_id,
            'user_id': user_id,
            'start_time': start_time,
            'duration': duration,
            'end_time': start_time + duration,
            'purpose': purpose,
            'status': 'confirmed'
        }
        
        # Add the reservation to our tracking
        self.reservations[reservation_id] = reservation
        
        # Update the resource status
        resource = self.hardware_resources[resource_id]
        resource.reserve(reservation_id)
        
        return True, f"Reservation {reservation_id} created successfully"
    
    def cancel_reservation(self, reservation_id: str) -> Tuple[bool, str]:
        """
        Cancel a hardware reservation
        
        Args:
            reservation_id: ID of the reservation to cancel
            
        Returns:
            Tuple of (success, message) where success is a boolean
            and message is a string with details
        """
        if reservation_id not in self.reservations:
            return False, f"Reservation {reservation_id} does not exist"
        
        reservation = self.reservations[reservation_id]
        resource_id = reservation['resource_id']
        
        # Move the reservation to history
        self.reservation_history.append(reservation)
        
        # Remove from active reservations
        del self.reservations[reservation_id]
        
        # Mark hardware as available again
        resource = self.hardware_resources[resource_id]
        resource.release_reservation()
        
        return True, f"Reservation {reservation_id} cancelled successfully"
    
    def get_reservation(self, reservation_id: str) -> Optional[Dict]:
        """Get details of a specific reservation"""
        return self.reservations.get(reservation_id)
    
    def get_user_reservations(self, user_id: str) -> List[Dict]:
        """Get all reservations for a specific user"""
        return [
            reservation for reservation in self.reservations.values()
            if reservation['user_id'] == user_id
        ]
    
    def get_resource_reservations(self, resource_id: str) -> List[Dict]:
        """Get all reservations for a specific resource"""
        return [
            reservation for reservation in self.reservations.values()
            if reservation['resource_id'] == resource_id
        ]
    
    def get_upcoming_reservations(self, resource_id: str, hours: int = 24) -> List[Dict]:
        """Get upcoming reservations for a resource within the specified time window"""
        now = datetime.now()
        future_time = now + timedelta(hours=hours)
        
        return [
            reservation for reservation in self.reservations.values()
            if (reservation['resource_id'] == resource_id and 
                reservation['start_time'] >= now and 
                reservation['start_time'] <= future_time)
        ]
    
    def update_hardware_status(self, resource_id: str, new_status: HardwareStatus) -> bool:
        """Update the status of a hardware resource"""
        if resource_id not in self.hardware_resources:
            return False
        
        resource = self.hardware_resources[resource_id]
        resource.status = new_status
        
        # If marking as available, clear any reservation
        if new_status == HardwareStatus.AVAILABLE:
            resource.release_reservation()
        
        # If marking as not available, cancel any active reservation
        elif new_status in [HardwareStatus.MAINTENANCE, HardwareStatus.FAULTY]:
            # Find and cancel any active reservation for this resource
            for res_id, res in list(self.reservations.items()):
                if res['resource_id'] == resource_id:
                    self.reservation_history.append(res)
                    del self.reservations[res_id]
        
        return True
    
    def get_resource_status(self, resource_id: str) -> Optional[HardwareStatus]:
        """Get the current status of a hardware resource"""
        resource = self.hardware_resources.get(resource_id)
        return resource.status if resource else None