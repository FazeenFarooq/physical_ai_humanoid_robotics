"""
Capstone project entity model for the Physical AI & Humanoid Robotics course.
This model represents the final project where students integrate all learned components.
Based on the data model specification in /specs/001-physical-ai-course/data-model.md
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class CapstoneMilestoneStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CapstoneProject:
    """
    Capstone project entity representing the final project for students.
    
    This project integrates all components learned throughout the course:
    - Voice-to-Intent (Milestone 1)
    - Perception & Mapping (Milestone 2)
    - Navigation & Obstacle Avoidance (Milestone 3)
    - Object Identification & Manipulation (Milestone 4)
    """
    id: str
    student_id: str  # Links to a Student entity
    milestone_1_status: CapstoneMilestoneStatus = CapstoneMilestoneStatus.NOT_STARTED  # Voice-to-Intent
    milestone_2_status: CapstoneMilestoneStatus = CapstoneMilestoneStatus.NOT_STARTED  # Perception & Mapping
    milestone_3_status: CapstoneMilestoneStatus = CapstoneMilestoneStatus.NOT_STARTED  # Navigation & Obstacle Avoidance
    milestone_4_status: CapstoneMilestoneStatus = CapstoneMilestoneStatus.NOT_STARTED  # Object Identification & Manipulation
    final_demo_status: CapstoneMilestoneStatus = CapstoneMilestoneStatus.NOT_STARTED  # Final demonstration
    components: List[str] = None  # List of integrated system components
    performance_metrics: Dict[str, float] = None  # Actual performance measurements
    failure_analysis: List[str] = None  # Analysis of failed attempts and lessons learned
    final_score: Optional[float] = None  # Overall evaluation score

    def __post_init__(self):
        """Initialize mutable default values"""
        if self.components is None:
            self.components = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.failure_analysis is None:
            self.failure_analysis = []

    def is_complete(self) -> bool:
        """Check if all milestones are completed"""
        return all([
            self.milestone_1_status == CapstoneMilestoneStatus.COMPLETED,
            self.milestone_2_status == CapstoneMilestoneStatus.COMPLETED,
            self.milestone_3_status == CapstoneMilestoneStatus.COMPLETED,
            self.milestone_4_status == CapstoneMilestoneStatus.COMPLETED,
            self.final_demo_status == CapstoneMilestoneStatus.COMPLETED
        ])

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage based on milestones"""
        statuses = [
            self.milestone_1_status,
            self.milestone_2_status,
            self.milestone_3_status,
            self.milestone_4_status,
            self.final_demo_status
        ]
        completed_count = sum(1 for status in statuses if status == CapstoneMilestoneStatus.COMPLETED)
        return (completed_count / len(statuses)) * 100