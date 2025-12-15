"""
Capstone project service for the Physical AI & Humanoid Robotics course.
Handles all business logic related to capstone project management.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Optional
from src.models.capstone_project import CapstoneProject, CapstoneMilestoneStatus
from src.models.student import Student


class CapstoneService:
    """Service class to manage capstone projects for students"""
    
    def __init__(self):
        self.capstone_projects: Dict[str, CapstoneProject] = {}
    
    def create_capstone_project(self, student: Student) -> CapstoneProject:
        """
        Create a new capstone project for a student
        """
        capstone_id = f"capstone_{student.id}"
        
        # Verify student has completed prerequisites for capstone
        if not self._student_meets_capstone_prerequisites(student):
            raise ValueError(f"Student {student.id} does not meet prerequisites for capstone project")
        
        capstone = CapstoneProject(
            id=capstone_id,
            student_id=student.id
        )
        
        self.capstone_projects[capstone_id] = capstone
        return capstone
    
    def get_capstone_project(self, capstone_id: str) -> Optional[CapstoneProject]:
        """
        Retrieve a capstone project by ID
        """
        return self.capstone_projects.get(capstone_id)
    
    def get_capstone_by_student(self, student_id: str) -> Optional[CapstoneProject]:
        """
        Retrieve a capstone project by student ID
        """
        for capstone in self.capstone_projects.values():
            if capstone.student_id == student_id:
                return capstone
        return None
    
    def update_milestone_status(self, capstone_id: str, milestone_number: int, status: CapstoneMilestoneStatus) -> bool:
        """
        Update the status of a specific milestone in a capstone project
        Milestone numbers: 1=Voice-to-Intent, 2=Perception & Mapping, 
        3=Navigation & Obstacle Avoidance, 4=Object Identification & Manipulation
        """
        capstone = self.capstone_projects.get(capstone_id)
        if not capstone:
            return False
        
        if milestone_number == 1:
            capstone.milestone_1_status = status
        elif milestone_number == 2:
            capstone.milestone_2_status = status
        elif milestone_number == 3:
            capstone.milestone_3_status = status
        elif milestone_number == 4:
            capstone.milestone_4_status = status
        else:
            return False
        
        return True
    
    def update_final_demo_status(self, capstone_id: str, status: CapstoneMilestoneStatus) -> bool:
        """
        Update the final demonstration status of a capstone project
        """
        capstone = self.capstone_projects.get(capstone_id)
        if not capstone:
            return False
        
        capstone.final_demo_status = status
        return True
    
    def add_performance_metric(self, capstone_id: str, metric_name: str, value: float) -> bool:
        """
        Add a performance metric to a capstone project
        """
        capstone = self.capstone_projects.get(capstone_id)
        if not capstone:
            return False
        
        capstone.performance_metrics[metric_name] = value
        return True
    
    def add_failure_analysis(self, capstone_id: str, analysis: str) -> bool:
        """
        Add failure analysis to a capstone project
        """
        capstone = self.capstone_projects.get(capstone_id)
        if not capstone:
            return False
        
        capstone.failure_analysis.append(analysis)
        return True
    
    def _student_meets_capstone_prerequisites(self, student: Student) -> bool:
        """
        Check if a student meets the prerequisites for starting a capstone project
        """
        # For now, we'll check if the student has completed at least 80% of the course
        # In a real implementation, we would check specific module completion requirements
        return student.progress >= 80.0
    
    def get_completion_report(self, capstone_id: str) -> Dict:
        """
        Get a detailed completion report for a capstone project
        """
        capstone = self.capstone_projects.get(capstone_id)
        if not capstone:
            return {}
        
        return {
            'id': capstone.id,
            'student_id': capstone.student_id,
            'milestone_1_status': capstone.milestone_1_status.value,
            'milestone_2_status': capstone.milestone_2_status.value,
            'milestone_3_status': capstone.milestone_3_status.value,
            'milestone_4_status': capstone.milestone_4_status.value,
            'final_demo_status': capstone.final_demo_status.value,
            'completion_percentage': capstone.get_completion_percentage(),
            'components_count': len(capstone.components),
            'performance_metrics_count': len(capstone.performance_metrics),
            'failure_analysis_count': len(capstone.failure_analysis),
            'is_complete': capstone.is_complete()
        }