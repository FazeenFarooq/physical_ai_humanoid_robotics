"""
Module entity model for the Physical AI & Humanoid Robotics course.
This model represents a self-contained course unit with objectives, inputs, and outputs.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Module:
    """Module entity representing a self-contained course unit"""
    id: str
    name: str
    description: str
    duration: int  # Estimated time to complete in weeks (1-4)
    prerequisites: List[str]  # Skills or modules required before starting
    objectives: List[str]  # List of learning objectives
    theory_topics: List[str]  # List of theoretical concepts covered
    lab_exercises: List[str]  # Collection of hands-on lab exercise IDs
    deliverables: List[str]  # Required outputs from the module
    toolchain: str  # Software and hardware tools used
    
    def is_available_to_student(self, student_prerequisites: List[str]) -> bool:
        """Check if a student has the prerequisites to start this module"""
        for prereq in self.prerequisites:
            if prereq not in student_prerequisites:
                return False
        return True

    def get_completion_percentage(self, completed_lab_exercises: List[str]) -> float:
        """Calculate the completion percentage based on completed lab exercises"""
        if not self.lab_exercises:
            return 100.0 if self.deliverables else 0.0
        
        completed_count = sum(1 for lab_id in self.lab_exercises if lab_id in completed_lab_exercises)
        return (completed_count / len(self.lab_exercises)) * 100.0