"""
Student tracking and management service for the Physical AI & Humanoid Robotics course.
This service provides functionality to manage students, modules, and lab exercises.
"""

from typing import Dict, List, Optional
from .entities import Student, Module, LabExercise, CapstoneProject, HardwareResource
import uuid
from datetime import datetime


class StudentManagementService:
    """Service to manage students, modules, and lab exercises"""
    
    def __init__(self):
        self.students: Dict[str, Student] = {}
        self.modules: Dict[str, Module] = {}
        self.lab_exercises: Dict[str, LabExercise] = {}
        self.capstone_projects: Dict[str, CapstoneProject] = {}
        self.hardware_resources: Dict[str, HardwareResource] = {}
    
    def create_student(self, name: str, email: str, prerequisites: List[str]) -> Student:
        """Create a new student in the system"""
        student_id = str(uuid.uuid4())
        student = Student(
            id=student_id,
            name=name,
            email=email,
            enrollment_date=datetime.now(),
            prerequisites=prerequisites
        )
        self.students[student_id] = student
        return student
    
    def get_student(self, student_id: str) -> Optional[Student]:
        """Retrieve a student by ID"""
        return self.students.get(student_id)
    
    def update_student_progress(self, student_id: str, module_id: str, progress: float) -> bool:
        """Update a student's progress in a specific module"""
        student = self.get_student(student_id)
        if student:
            student.current_module = module_id
            student.progress = progress
            return True
        return False
    
    def create_module(self, name: str, description: str, prerequisites: List[str], 
                     objectives: List[str], theory_topics: List[str], 
                     lab_exercises: List[str], deliverables: List[str], 
                     toolchain: str, duration: int = 1) -> Module:
        """Create a new module in the system"""
        module_id = str(uuid.uuid4())
        module = Module(
            id=module_id,
            name=name,
            description=description,
            duration=duration,
            prerequisites=prerequisites,
            objectives=objectives,
            theory_topics=theory_topics,
            lab_exercises=lab_exercises,
            deliverables=deliverables,
            toolchain=toolchain
        )
        self.modules[module_id] = module
        return module
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Retrieve a module by ID"""
        return self.modules.get(module_id)
    
    def create_lab_exercise(self, name: str, description: str, objectives: List[str],
                           steps: List[str], validation_criteria: List[str],
                           resources: List[str], difficulty: int, estimated_time: int,
                           related_module: str) -> LabExercise:
        """Create a new lab exercise in the system"""
        exercise_id = str(uuid.uuid4())
        exercise = LabExercise(
            id=exercise_id,
            name=name,
            description=description,
            objectives=objectives,
            steps=steps,
            validation_criteria=validation_criteria,
            resources=resources,
            difficulty=difficulty,
            estimated_time=estimated_time,
            related_module=related_module
        )
        self.lab_exercises[exercise_id] = exercise
        return exercise
    
    def get_lab_exercise(self, exercise_id: str) -> Optional[LabExercise]:
        """Retrieve a lab exercise by ID"""
        return self.lab_exercises.get(exercise_id)
    
    def create_capstone_project(self, student_id: str) -> CapstoneProject:
        """Create a new capstone project for a student"""
        capstone_id = str(uuid.uuid4())
        capstone = CapstoneProject(
            id=capstone_id,
            student_id=student_id
        )
        self.capstone_projects[capstone_id] = capstone
        return capstone
    
    def get_capstone_project(self, student_id: str) -> Optional[CapstoneProject]:
        """Retrieve a student's capstone project"""
        for capstone in self.capstone_projects.values():
            if capstone.student_id == student_id:
                return capstone
        return None
    
    def create_hardware_resource(self, type: str, model: str, location: str) -> HardwareResource:
        """Create a new hardware resource in the system"""
        resource_id = str(uuid.uuid4())
        resource = HardwareResource(
            id=resource_id,
            type=type,
            model=model,
            location=location
        )
        self.hardware_resources[resource_id] = resource
        return resource
    
    def get_hardware_resource(self, resource_id: str) -> Optional[HardwareResource]:
        """Retrieve a hardware resource by ID"""
        return self.hardware_resources.get(resource_id)
    
    def reserve_hardware(self, resource_id: str, student_id: str) -> bool:
        """Reserve a hardware resource for a student"""
        resource = self.get_hardware_resource(resource_id)
        if resource and resource.status == "available":
            resource.status = "reserved"
            resource.reservation = student_id
            return True
        return False
    
    def get_available_hardware(self, hardware_type: str) -> List[HardwareResource]:
        """Get all available hardware resources of a specific type"""
        available = []
        for resource in self.hardware_resources.values():
            if resource.type == hardware_type and resource.status == "available":
                available.append(resource)
        return available