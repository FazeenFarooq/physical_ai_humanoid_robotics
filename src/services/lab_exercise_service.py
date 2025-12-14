"""
LabExercise creation and management service for the Physical AI & Humanoid Robotics course.
This service handles creation, retrieval, and management of lab exercises.
"""

from typing import Dict, List, Optional
from ..models.lab_exercise import LabExercise, LabExerciseStatus
from ..models.module import Module
import uuid
from datetime import datetime


class LabExerciseService:
    """Service to manage lab exercises"""
    
    def __init__(self):
        self.lab_exercises: Dict[str, LabExercise] = {}
    
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
    
    def get_all_lab_exercises(self) -> List[LabExercise]:
        """Retrieve all lab exercises"""
        return list(self.lab_exercises.values())
    
    def get_lab_exercises_by_module(self, module_id: str) -> List[LabExercise]:
        """Get all lab exercises for a specific module"""
        return [lab for lab in self.lab_exercises.values() if lab.related_module == module_id]
    
    def update_lab_exercise(self, exercise_id: str, **kwargs) -> bool:
        """Update a lab exercise's properties"""
        lab_exercise = self.get_lab_exercise(exercise_id)
        if not lab_exercise:
            return False
            
        # Update only the provided fields
        for key, value in kwargs.items():
            if hasattr(lab_exercise, key):
                setattr(lab_exercise, key, value)
        
        return True
    
    def update_lab_status(self, exercise_id: str, status: LabExerciseStatus) -> bool:
        """Update the status of a lab exercise"""
        lab_exercise = self.get_lab_exercise(exercise_id)
        if lab_exercise:
            lab_exercise.status = status
            return True
        return False
    
    def validate_submission(self, exercise_id: str, submission: dict) -> dict:
        """Validate a student's lab submission against criteria"""
        lab_exercise = self.get_lab_exercise(exercise_id)
        if not lab_exercise:
            return {"valid": False, "error": "Lab exercise not found"}
        
        # This is a simplified validation - in a real system this would be more complex
        # For now, we'll just check if submission has required keys based on validation_criteria
        required_keys = []
        for criterion in lab_exercise.validation_criteria:
            # Extract required keys from validation criteria (this is simplified)
            if "result" in criterion.lower():
                required_keys.append("result")
            elif "output" in criterion.lower():
                required_keys.append("output")
        
        missing_keys = [key for key in required_keys if key not in submission]
        
        if missing_keys:
            return {
                "valid": False, 
                "error": f"Missing required components: {missing_keys}",
                "required": required_keys
            }
        
        # If all required keys are present, mark as valid
        score = lab_exercise.get_validation_score(submission)
        return {
            "valid": True,
            "score": score,
            "message": "Submission meets all validation criteria"
        }
    
    def get_exercises_by_difficulty(self, min_difficulty: int, max_difficulty: int) -> List[LabExercise]:
        """Get lab exercises within a difficulty range"""
        return [
            lab for lab in self.lab_exercises.values()
            if min_difficulty <= lab.difficulty <= max_difficulty
        ]
    
    def get_average_completion_time(self, exercise_id: str) -> Optional[float]:
        """Calculate average completion time for an exercise (would need submission history)"""
        # This would require tracking submission history to calculate averages
        # For now, return None
        return None