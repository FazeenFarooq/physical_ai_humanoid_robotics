"""
PerceptionData Entity Model for the Physical AI & Humanoid Robotics Course.

This module defines the PerceptionData entity, which represents sensor data collected 
by the robotic system for processing by vision-language-action (VLA) models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import numpy as np


@dataclass
class PerceptionData:
    """Entity representing perception data collected by the robot."""
    
    # Unique identifier for this perception data
    id: str
    # Timestamp when data was captured
    timestamp: datetime
    # Type of sensor (RGB-D, LiDAR, IMU, Audio, etc.)
    sensor_type: str
    # Raw sensor data (format depends on sensor_type)
    data: bytes
    # Environment where data was collected
    environment: str
    # Robot's state when data was collected
    robot_state: Dict[str, Any]
    # Annotations or labels for the data
    annotations: List[Dict[str, Any]]
    # Source: Simulation or physical robot
    source: str  # 'simulation' or 'physical'
    # Quality score for the data
    quality_score: float = 1.0
    # Processing status
    processing_status: str = 'raw'  # 'raw', 'processed', 'annotated', 'validated'
    # Associated action command
    action_command_id: Optional[str] = None
    # Confidence in the data
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate the PerceptionData after initialization."""
        if self.quality_score < 0.0 or self.quality_score > 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
        
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        
        if self.source not in ['simulation', 'physical']:
            raise ValueError("source must be 'simulation' or 'physical'")
        
        if self.processing_status not in ['raw', 'processed', 'annotated', 'validated']:
            raise ValueError("processing_status must be one of 'raw', 'processed', 'annotated', 'validated'")


@dataclass
class PerceptionDataStats:
    """Statistics for perception data."""
    
    total_count: int
    by_sensor_type: Dict[str, int]
    by_source: Dict[str, int]
    avg_quality_score: float
    avg_confidence: float
    date_range: tuple  # (min_date, max_date)


class PerceptionDataManager:
    """Manager for PerceptionData entities."""
    
    def __init__(self):
        self.perception_data: Dict[str, PerceptionData] = {}
        self.annotation_templates: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_perception_data(self, perception_data: PerceptionData) -> bool:
        """Add perception data to the manager."""
        if perception_data.id in self.perception_data:
            return False
        
        self.perception_data[perception_data.id] = perception_data
        return True
    
    def get_perception_data(self, data_id: str) -> Optional[PerceptionData]:
        """Retrieve perception data by ID."""
        return self.perception_data.get(data_id)
    
    def get_perception_data_by_sensor_type(self, sensor_type: str) -> List[PerceptionData]:
        """Get all perception data of a certain sensor type."""
        return [data for data in self.perception_data.values() if data.sensor_type == sensor_type]
    
    def get_perception_data_by_source(self, source: str) -> List[PerceptionData]:
        """Get all perception data from a certain source."""
        return [data for data in self.perception_data.values() if data.source == source]
    
    def update_processing_status(self, data_id: str, status: str) -> bool:
        """Update the processing status of perception data."""
        if data_id not in self.perception_data:
            return False
        
        if status not in ['raw', 'processed', 'annotated', 'validated']:
            return False
        
        self.perception_data[data_id].processing_status = status
        return True
    
    def get_stats(self) -> PerceptionDataStats:
        """Get statistics about the perception data."""
        if not self.perception_data:
            return PerceptionDataStats(
                total_count=0,
                by_sensor_type={},
                by_source={},
                avg_quality_score=0.0,
                avg_confidence=0.0,
                date_range=(datetime.now(), datetime.now())
            )
        
        total_count = len(self.perception_data)
        
        # Count by sensor type
        by_sensor_type = {}
        for data in self.perception_data.values():
            sensor_type = data.sensor_type
            by_sensor_type[sensor_type] = by_sensor_type.get(sensor_type, 0) + 1
        
        # Count by source
        by_source = {}
        for data in self.perception_data.values():
            source = data.source
            by_source[source] = by_source.get(source, 0) + 1
        
        # Calculate average quality and confidence
        avg_quality_score = sum(data.quality_score for data in self.perception_data.values()) / total_count
        avg_confidence = sum(data.confidence for data in self.perception_data.values()) / total_count
        
        # Determine date range
        timestamps = [data.timestamp for data in self.perception_data.values()]
        date_range = (min(timestamps), max(timestamps))
        
        return PerceptionDataStats(
            total_count=total_count,
            by_sensor_type=by_sensor_type,
            by_source=by_source,
            avg_quality_score=avg_quality_score,
            avg_confidence=avg_confidence,
            date_range=date_range
        )
    
    def validate_perception_data(self, data_id: str) -> bool:
        """Validate perception data."""
        data = self.get_perception_data(data_id)
        if not data:
            return False
        
        # Perform validation checks
        # For now, just update status to validated
        self.update_processing_status(data_id, 'validated')
        return True
    
    def create_annotation_template(self, template_name: str, annotation_schema: List[Dict[str, Any]]):
        """Create an annotation template."""
        self.annotation_templates[template_name] = annotation_schema
    
    def get_annotation_template(self, template_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get an annotation template."""
        return self.annotation_templates.get(template_name)


# Example usage and creation of the entity
def create_example_perception_data() -> PerceptionData:
    """Create an example PerceptionData instance."""
    return PerceptionData(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        sensor_type="RGB-D",
        data=b"example_binary_data",
        environment="indoor_lab",
        robot_state={
            "position": {"x": 1.5, "y": 2.0, "z": 0.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.2},
            "joints": {"head_pan": 0.1, "head_tilt": 0.05}
        },
        annotations=[
            {
                "type": "object_detection",
                "label": "cup",
                "confidence": 0.95,
                "bbox": [100, 150, 200, 250]  # [x1, y1, x2, y2]
            }
        ],
        source="simulation",
        quality_score=0.9,
        confidence=0.95
    )


if __name__ == "__main__":
    # Example usage
    manager = PerceptionDataManager()
    
    # Create and add example perception data
    example_data = create_example_perception_data()
    manager.add_perception_data(example_data)
    
    print(f"Added perception data with ID: {example_data.id}")
    print(f"Sensor type: {example_data.sensor_type}")
    print(f"Source: {example_data.source}")
    print(f"Quality score: {example_data.quality_score}")
    
    # Get statistics
    stats = manager.get_stats()
    print(f"\nPerception Data Statistics:")
    print(f"Total count: {stats.total_count}")
    print(f"By sensor type: {stats.by_sensor_type}")
    print(f"By source: {stats.by_source}")
    print(f"Average quality score: {stats.avg_quality_score:.2f}")