"""
Student progress tracking service for the Physical AI & Humanoid Robotics course.
This service handles tracking and reporting of student progress across modules and lab exercises.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..models.student import Student
from ..models.module import Module
from ..models.lab_exercise import LabExercise
import uuid


class StudentProgressService:
    """Service to track and manage student progress"""
    
    def __init__(self):
        self.progress_records: Dict[str, Dict] = {}  # Maps student_id to progress data
        self.completion_history: List[Dict] = []  # Track completion events
    
    def initialize_student_progress(self, student_id: str, student_name: str) -> bool:
        """Initialize progress tracking for a new student"""
        if student_id not in self.progress_records:
            self.progress_records[student_id] = {
                'student_id': student_id,
                'student_name': student_name,
                'enrollment_date': datetime.now(),
                'modules_completed': [],
                'lab_exercises_completed': [],
                'current_module': None,
                'progress_percentage': 0.0,
                'time_spent': timedelta(0),  # Total time spent in course
                'last_activity': datetime.now()
            }
            return True
        return False
    
    def record_lab_completion(self, student_id: str, lab_id: str, module_id: str, 
                             score: Optional[float] = None) -> bool:
        """Record completion of a lab exercise by a student"""
        if student_id not in self.progress_records:
            return False
        
        record = self.progress_records[student_id]
        
        # Add to completed labs if not already there
        if lab_id not in record['lab_exercises_completed']:
            record['lab_exercises_completed'].append(lab_id)
            
            # Add to completion history
            self.completion_history.append({
                'student_id': student_id,
                'item_type': 'lab_exercise',
                'item_id': lab_id,
                'module_id': module_id,
                'completed_at': datetime.now(),
                'score': score
            })
            
            # Update progress percentage
            self._update_progress_percentage(student_id)
            record['last_activity'] = datetime.now()
            return True
        
        return False
    
    def record_module_completion(self, student_id: str, module_id: str) -> bool:
        """Record completion of a module by a student"""
        if student_id not in self.progress_records:
            return False
        
        record = self.progress_records[student_id]
        
        if module_id not in record['modules_completed']:
            record['modules_completed'].append(module_id)
            
            # Add to completion history
            self.completion_history.append({
                'student_id': student_id,
                'item_type': 'module',
                'item_id': module_id,
                'completed_at': datetime.now()
            })
            
            # Update current module
            record['current_module'] = module_id
            
            # Update progress percentage
            self._update_progress_percentage(student_id)
            record['last_activity'] = datetime.now()
            return True
        
        return False
    
    def _update_progress_percentage(self, student_id: str):
        """Update the overall progress percentage for a student"""
        if student_id not in self.progress_records:
            return
        
        # This is a simplified calculation - in a real system this would be more complex
        # based on the actual course structure and requirements
        record = self.progress_records[student_id]
        
        # For now, we'll calculate based on modules and labs completed
        # This would be customized based on course requirements
        total_modules = 6  # Based on the 6 modules in the course
        completed_modules = len(record['modules_completed'])
        
        # We'll assign a weight of 70% to modules and 30% to lab exercises
        module_percentage = (completed_modules / total_modules) * 70 if total_modules > 0 else 0
        
        # For lab exercises, we'll estimate the number of labs per module
        # For simplicity, assume an average number of labs
        estimated_total_labs = 30  # Estimated total labs in course
        completed_labs = len(record['lab_exercises_completed'])
        lab_percentage = (completed_labs / estimated_total_labs) * 30 if estimated_total_labs > 0 else 0
        
        record['progress_percentage'] = min(100.0, module_percentage + lab_percentage)
    
    def get_student_progress(self, student_id: str) -> Optional[Dict]:
        """Get progress information for a specific student"""
        return self.progress_records.get(student_id)
    
    def get_all_student_progress(self) -> List[Dict]:
        """Get progress information for all students"""
        return list(self.progress_records.values())
    
    def get_completion_percentage(self, student_id: str) -> float:
        """Get overall completion percentage for a student"""
        record = self.progress_records.get(student_id)
        if record:
            return record['progress_percentage']
        return 0.0
    
    def get_labs_for_student(self, student_id: str) -> List[str]:
        """Get list of completed lab exercises for a student"""
        record = self.progress_records.get(student_id)
        if record:
            return record['lab_exercises_completed']
        return []
    
    def get_modules_for_student(self, student_id: str) -> List[str]:
        """Get list of completed modules for a student"""
        record = self.progress_records.get(student_id)
        if record:
            return record['modules_completed']
        return []
    
    def update_time_spent(self, student_id: str, time_delta: timedelta) -> bool:
        """Update the time spent tracking for a student"""
        if student_id not in self.progress_records:
            return False
        
        record = self.progress_records[student_id]
        record['time_spent'] += time_delta
        record['last_activity'] = datetime.now()
        return True
    
    def get_completion_history(self, student_id: Optional[str] = None) -> List[Dict]:
        """Get completion history, optionally filtered by student"""
        if student_id:
            return [r for r in self.completion_history if r['student_id'] == student_id]
        return self.completion_history
    
    def get_progress_report(self, student_id: str) -> Dict:
        """Generate a comprehensive progress report for a student"""
        record = self.get_student_progress(student_id)
        if not record:
            return {}
        
        # Get completion history for this student
        history = self.get_completion_history(student_id)
        
        # Calculate statistics
        lab_completions = [item for item in history if item['item_type'] == 'lab_exercise']
        module_completions = [item for item in history if item['item_type'] == 'module']
        
        avg_lab_score = None
        if lab_completions:
            valid_scores = [item['score'] for item in lab_completions if item['score'] is not None]
            if valid_scores:
                avg_lab_score = sum(valid_scores) / len(valid_scores)
        
        return {
            'student_id': record['student_id'],
            'student_name': record['student_name'],
            'overall_progress': record['progress_percentage'],
            'modules_completed': len(record['modules_completed']),
            'labs_completed': len(record['lab_exercises_completed']),
            'time_spent': str(record['time_spent']),
            'last_activity': record['last_activity'],
            'enrollment_date': record['enrollment_date'],
            'current_module': record['current_module'],
            'average_lab_score': avg_lab_score,
            'total_activities_completed': len(history)
        }
    
    def reset_student_progress(self, student_id: str) -> bool:
        """Reset a student's progress (useful for re-enrollment)"""
        if student_id in self.progress_records:
            # Keep basic info but reset progress
            basic_info = {
                'student_id': self.progress_records[student_id]['student_id'],
                'student_name': self.progress_records[student_id]['student_name'],
                'enrollment_date': self.progress_records[student_id]['enrollment_date'],
            }
            
            # Initialize with defaults
            self.initialize_student_progress(student_id, basic_info['student_name'])
            
            # Restore basic info
            record = self.progress_records[student_id]
            record['enrollment_date'] = basic_info['enrollment_date']
            
            return True
        return False