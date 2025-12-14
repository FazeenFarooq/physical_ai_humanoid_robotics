"""
Module management service for the Physical AI & Humanoid Robotics course.
This service handles creation, retrieval, and management of course modules.
"""

from typing import Dict, List, Optional
from ..models.module import Module
from ..models.student import Student
import uuid
from datetime import datetime


class ModuleService:
    """Service to manage course modules"""
    
    def __init__(self):
        self.modules: Dict[str, Module] = {}
    
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
    
    def get_all_modules(self) -> List[Module]:
        """Retrieve all modules"""
        return list(self.modules.values())
    
    def update_module(self, module_id: str, **kwargs) -> bool:
        """Update a module's properties"""
        module = self.get_module(module_id)
        if not module:
            return False
            
        # Update only the provided fields
        for key, value in kwargs.items():
            if hasattr(module, key):
                setattr(module, key, value)
        
        return True
    
    def get_available_modules_for_student(self, student: Student) -> List[Module]:
        """Get modules that a student is eligible to take based on prerequisites"""
        available_modules = []
        for module in self.modules.values():
            if module.is_available_to_student(student.prerequisites):
                available_modules.append(module)
        return available_modules
    
    def get_module_completion_percentage(self, module_id: str, completed_lab_exercises: List[str]) -> float:
        """Get completion percentage for a specific module"""
        module = self.get_module(module_id)
        if not module:
            return 0.0
        
        return module.get_completion_percentage(completed_lab_exercises)
    
    def add_lab_to_module(self, module_id: str, lab_id: str) -> bool:
        """Add a lab exercise to a module"""
        module = self.get_module(module_id)
        if module and lab_id not in module.lab_exercises:
            module.lab_exercises.append(lab_id)
            return True
        return False
    
    def remove_lab_from_module(self, module_id: str, lab_id: str) -> bool:
        """Remove a lab exercise from a module"""
        module = self.get_module(module_id)
        if module and lab_id in module.lab_exercises:
            module.lab_exercises.remove(lab_id)
            return True
        return False