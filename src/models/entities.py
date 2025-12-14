"""
Student tracking and management system entities for the Physical AI & Humanoid Robotics course.
These models represent the core entities described in the data model specification.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum


class LabExerciseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class HardwareStatus(Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    FAULTY = "faulty"


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
    
    def is_available_to_student(self, student: Student) -> bool:
        """Check if a student has the prerequisites to start this module"""
        for prereq in self.prerequisites:
            if prereq not in student.prerequisites:
                return False
        return True


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


@dataclass
class CapstoneProject:
    """Capstone project entity representing the final project"""
    id: str
    student_id: str
    milestone_1_status: str = "not_started"  # Voice-to-Intent
    milestone_2_status: str = "not_started"  # Perception & Mapping
    milestone_3_status: str = "not_started"  # Navigation & Obstacle Avoidance
    milestone_4_status: str = "not_started"  # Object Identification & Manipulation
    final_demo_status: str = "not_started"  # Final demonstration
    components: List[str] = None  # List of integrated system components
    performance_metrics: dict = None  # Actual performance measurements
    failure_analysis: List[str] = None  # Analysis of failed attempts and lessons learned
    final_score: Optional[float] = None  # Overall evaluation score
    
    def __post_init__(self):
        if self.components is None:
            self.components = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.failure_analysis is None:
            self.failure_analysis = []


@dataclass
class HardwareResource:
    """Hardware resource entity for tracking course equipment"""
    id: str
    type: str  # Workstation, Jetson Orin, Robot Platform, etc.
    model: str  # Specific model and specifications
    location: str  # Physical location of the hardware
    status: HardwareStatus = HardwareStatus.AVAILABLE
    reservation: Optional[str] = None  # Current reservation information
    owner: Optional[str] = None  # Person responsible for maintenance
    last_calibration: Optional[datetime] = None  # Date of last calibration
    availability_schedule: Optional[str] = None  # When the hardware is available