"""
Student entity model for the Physical AI & Humanoid Robotics course.
This model represents a course participant with specific prerequisites.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Student:
    """Student entity representing a course participant"""
    id: str
    name: str
    email: str
    enrollment_date: datetime
    prerequisites: List[str]  # List of verified prerequisites (Python, ML, Linux, etc.)
    current_module: Optional[str] = None
    progress: float = 0.0  # Percentage completion of course
    lab_submissions: List[str] = None  # List of completed lab assignment IDs
    capstone_status: str = "not_started"  # Current status of capstone project
    
    def __post_init__(self):
        if self.lab_submissions is None:
            self.lab_submissions = []

    def can_enroll_in_module(self, module_prerequisites: List[str]) -> bool:
        """Check if the student meets the prerequisites for a given module"""
        for prereq in module_prerequisites:
            if prereq not in self.prerequisites:
                return False
        return True

    def get_module_completion_percentage(self, module_lab_exercises: List[str]) -> float:
        """Calculate completion percentage for a specific module"""
        if not module_lab_exercises:
            return 0.0
            
        completed_count = sum(1 for lab_id in module_lab_exercises if lab_id in self.lab_submissions)
        return (completed_count / len(module_lab_exercises)) * 100.0

    def add_lab_submission(self, lab_id: str) -> bool:
        """Add a completed lab to the student's record"""
        if lab_id not in self.lab_submissions:
            self.lab_submissions.append(lab_id)
            return True
        return False