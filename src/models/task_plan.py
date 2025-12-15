"""
Task plan entity model for the Physical AI & Humanoid Robotics course.
This model represents high-level tasks that the robot needs to execute.
Based on the data model specification in /specs/001-physical-ai-course/data-model.md
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    """Status of a task plan"""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority level for task plans"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskStep:
    """Represents a single step in a task plan"""
    id: str
    description: str
    action_type: str  # e.g., "navigation", "manipulation", "perception", "conversation"
    parameters: Dict[str, Any]
    estimated_duration: float  # in seconds
    dependencies: List[str]  # IDs of steps this step depends on
    success_criteria: List[str]  # Conditions for step completion
    failure_recovery: Optional[str]  # What to do if this step fails


@dataclass
class TaskPlan:
    """
    TaskPlan entity representing a high-level task for the robot.
    This includes natural language description, execution steps,
    dependencies, and success criteria.
    """
    id: str
    name: str
    description: str  # Natural language description of the task
    steps: List[TaskStep]  # Sequential steps to complete the task
    requirements: List[str]  # Resources and conditions needed
    constraints: List[str]  # Limitations or restrictions
    success_criteria: List[str]  # How to determine task completion
    fallback_behaviors: List[str]  # Actions to take if primary approach fails
    estimated_time: float  # Estimated time for completion in seconds
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NOT_STARTED
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step_index: int = -1  # Index of currently executing step
    execution_history: List[Dict[str, Any]] = None  # Track execution events
    
    def __post_init__(self):
        """Initialize default values"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.execution_history is None:
            self.execution_history = []
    
    def get_next_step(self) -> Optional[TaskStep]:
        """Get the next step to execute"""
        if self.current_step_index + 1 < len(self.steps):
            return self.steps[self.current_step_index + 1]
        return None
    
    def advance_to_next_step(self) -> bool:
        """Move to the next step in the task plan"""
        if self.current_step_index + 1 < len(self.steps):
            self.current_step_index += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all steps have been completed"""
        return self.current_step_index >= len(self.steps) - 1 and self.status == TaskStatus.COMPLETED
    
    def can_start_step(self, step_index: int) -> bool:
        """Check if a step can be started (dependencies met)"""
        if step_index < 0 or step_index >= len(self.steps):
            return False
        
        step = self.steps[step_index]
        for dep_id in step.dependencies:
            # Check if dependency step is completed
            dep_idx = next((i for i, s in enumerate(self.steps) if s.id == dep_id), -1)
            if dep_idx == -1 or dep_idx > self.current_step_index:
                return False
        
        return True
    
    def get_progress(self) -> float:
        """Calculate completion percentage"""
        if not self.steps:
            return 0.0
        return min(1.0, max(0.0, (self.current_step_index + 1) / len(self.steps)))
    
    def add_execution_event(self, event: Dict[str, Any]):
        """Add an event to the execution history"""
        self.execution_history.append({
            "timestamp": datetime.now(),
            **event
        })
    
    def get_remaining_steps(self) -> List[TaskStep]:
        """Get steps that haven't been completed yet"""
        if self.current_step_index < 0:
            return self.steps[:]
        return self.steps[self.current_step_index + 1:]
    
    def get_completed_steps(self) -> List[TaskStep]:
        """Get steps that have been completed"""
        if self.current_step_index < 0:
            return []
        return self.steps[:self.current_step_index + 1]