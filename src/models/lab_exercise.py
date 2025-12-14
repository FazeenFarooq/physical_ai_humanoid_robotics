"""
LabExercise entity model for the Physical AI & Humanoid Robotics course.
This model represents a hands-on lab exercise for students.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime


class LabExerciseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LabExercise:
    """Lab exercise entity for hands-on learning activities"""
    id: str
    name: str
    description: str
    objectives: List[str]  # Learning objectives for this lab
    steps: List[str]  # Sequential steps to complete the lab
    validation_criteria: List[str]  # How to verify successful completion
    resources: List[str]  # Required files, models, or environments
    difficulty: int  # Level of complexity (1-5)
    estimated_time: int  # Time required in hours
    related_module: str  # Module this lab belongs to
    status: LabExerciseStatus = LabExerciseStatus.NOT_STARTED
    created_at: datetime = datetime.now()
    
    def is_valid_for_completion(self, student_progress: dict) -> bool:
        """Check if a student has the prerequisites to complete this lab"""
        # This would check if student has completed prerequisite modules
        return True  # Simplified for now
    
    def get_validation_score(self, submission: dict) -> float:
        """Calculate validation score based on submission"""
        score = 0.0
        total_criteria = len(self.validation_criteria)
        
        if total_criteria == 0:
            return 100.0  # If no criteria, consider complete
            
        # This would evaluate the submission against validation criteria
        # For now, return a default value
        return score