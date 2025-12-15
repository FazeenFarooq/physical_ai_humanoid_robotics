"""
Action command entity model for the Physical AI & Humanoid Robotics course.
This model represents specific actions that the robot should execute.
Based on the data model specification in /specs/001-physical-ai-course/data-model.md
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from enum import Enum
from datetime import datetime


class ActionType(Enum):
    """Types of actions that can be commanded"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    CONVERSATION = "conversation"
    SYSTEM_CONTROL = "system_control"
    DATA_COLLECTION = "data_collection"


class ActionStatus(Enum):
    """Status of an action command"""
    QUEUED = "queued"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ActionCommand:
    """
    ActionCommand entity representing a specific action for the robot to execute.
    This includes the action type, parameters, priority, and execution status.
    """
    id: str
    type: ActionType
    parameters: Dict[str, Any]  # Action-specific parameters
    priority: int = 5  # Priority level (0-10, where 10 is highest)
    status: ActionStatus = ActionStatus.QUEUED
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    executor: Optional[str] = None  # Component responsible for execution
    dependencies: List[str] = None  # Other action IDs that must complete first
    timeout: Optional[float] = 30.0  # Timeout in seconds
    retries: int = 0  # Number of retry attempts
    max_retries: int = 3  # Maximum number of retry attempts
    parent_task_id: Optional[str] = None  # ID of the parent task plan
    result: Optional[Dict[str, Any]] = None  # Execution result
    error_message: Optional[str] = None  # Error message if failed
    progress: float = 0.0  # Progress percentage (0.0 to 1.0)
    estimated_duration: Optional[float] = None  # Estimated execution time in seconds
    actual_duration: Optional[float] = None  # Actual execution time in seconds
    
    def __post_init__(self):
        """Initialize default values"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.result is None:
            self.result = {}
    
    def can_execute(self) -> bool:
        """Check if this action is ready to execute (dependencies satisfied)"""
        # This would typically check with a scheduler or dependency manager
        # For now, we'll assume dependencies are satisfied if they're in a completed state
        return self.status == ActionStatus.QUEUED
    
    def start_execution(self, executor_name: str = None):
        """Mark the action as started"""
        self.status = ActionStatus.EXECUTING
        self.started_at = datetime.now()
        if executor_name:
            self.executor = executor_name
    
    def complete_execution(self, success: bool = True, result: Dict[str, Any] = None):
        """Mark the action as completed"""
        self.completed_at = datetime.now()
        self.status = ActionStatus.COMPLETED if success else ActionStatus.FAILED
        if result:
            self.result = result
        # Calculate actual duration
        if self.started_at:
            self.actual_duration = (self.completed_at - self.started_at).total_seconds()
    
    def fail_execution(self, error_msg: str = None):
        """Mark the action as failed"""
        self.status = ActionStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_msg or "Action failed without specific error"
        # Calculate actual duration
        if self.started_at:
            self.actual_duration = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self, progress: float):
        """Update the progress of the action"""
        self.progress = max(0.0, min(1.0, progress))
    
    def update_status(self, new_status: ActionStatus):
        """Update the status of the action"""
        self.status = new_status
    
    def is_active(self) -> bool:
        """Check if the action is currently active (executing or queued)"""
        return self.status in [ActionStatus.QUEUED, ActionStatus.SCHEDULING, ActionStatus.EXECUTING]
    
    def is_finished(self) -> bool:
        """Check if the action is finished (completed, failed, or cancelled)"""
        return self.status in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED]
    
    def requires_attention(self) -> bool:
        """Check if the action requires attention (failed, timed out, etc.)"""
        return self.status in [ActionStatus.FAILED, ActionStatus.CANCELLED]
    
    def get_duration_estimate_accuracy(self) -> Optional[float]:
        """Calculate how accurate the duration estimate was"""
        if self.estimated_duration and self.actual_duration:
            return abs(self.estimated_duration - self.actual_duration) / self.estimated_duration
        return None


@dataclass
class ActionQueue:
    """Queue of action commands waiting for execution"""
    name: str
    actions: List[ActionCommand]
    max_size: Optional[int] = None
    priority_enabled: bool = True
    
    def enqueue(self, action: ActionCommand):
        """Add an action to the queue"""
        if self.max_size and len(self.actions) >= self.max_size:
            # Remove the lowest priority action if queue is full
            if self.priority_enabled:
                lowest_priority_idx = min(range(len(self.actions)), 
                                        key=lambda i: self.actions[i].priority)
                self.actions.pop(lowest_priority_idx)
        
        # Insert action based on priority
        if self.priority_enabled:
            # Simple insertion based on priority (higher priority first)
            inserted = False
            for i, existing_action in enumerate(self.actions):
                if action.priority > existing_action.priority:
                    self.actions.insert(i, action)
                    inserted = True
                    break
            
            if not inserted:
                self.actions.append(action)
        else:
            self.actions.append(action)
    
    def dequeue(self) -> Optional[ActionCommand]:
        """Remove and return the next action from the queue"""
        if self.actions:
            return self.actions.pop(0)
        return None
    
    def peek(self) -> Optional[ActionCommand]:
        """Return the next action without removing it"""
        if self.actions:
            return self.actions[0]
        return None
    
    def remove_by_id(self, action_id: str) -> bool:
        """Remove an action from the queue by ID"""
        for i, action in enumerate(self.actions):
            if action.id == action_id:
                self.actions.pop(i)
                return True
        return False
    
    def get_by_id(self, action_id: str) -> Optional[ActionCommand]:
        """Get an action from the queue by ID"""
        for action in self.actions:
            if action.id == action_id:
                return action
        return None
    
    def get_pending_actions(self) -> List[ActionCommand]:
        """Get all pending (not started) actions"""
        return [action for action in self.actions if not action.is_active()]
    
    def get_active_actions(self) -> List[ActionCommand]:
        """Get all active (executing) actions"""
        return [action for action in self.actions if action.is_active()]
    
    def clear(self):
        """Clear all actions from the queue"""
        self.actions = []