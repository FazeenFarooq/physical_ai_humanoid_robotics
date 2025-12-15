"""
Conversation stack for the capstone project in the Physical AI & Humanoid Robotics course.
This module integrates speech recognition, natural language understanding, 
dialogue management, and response generation for the complete conversational system.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
from enum import Enum
from datetime import datetime

from src.conversation.speech_recognition import ASRManager, SpeechRecognitionResult
from src.conversation.dialogue_manager import DialogueManager, DialogueResponse, IntentType
from src.conversation.nlg import NLGEngine
from src.conversation.llm_integration import ROS2LLMInterface, NIMClient
from src.conversation.safety_filters import SafetyManager


class ConversationStatus(Enum):
    """Status of conversation"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_INPUT = "waiting_for_input"
    ERROR = "error"


class InteractionMode(Enum):
    """Mode of interaction"""
    STANDALONE = "standalone"
    TASK_DIRECTED = "task_directed"
    SOCIAL = "social"
    INFORMATION_SEEKING = "information_seeking"


@dataclass
class ConversationContext:
    """Context information for the conversation"""
    user_id: str
    session_id: str
    current_mode: InteractionMode
    conversation_history: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    task_context: Optional[Dict[str, Any]]
    last_interaction_time: datetime
    interruption_level: int  # 0 = no interruption, 5 = high priority


@dataclass
class ConversationResult:
    """Result of a conversation turn"""
    success: bool
    user_input: str
    system_response: str
    intent: Optional[IntentType]
    entities: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class CapstoneConversationEngine:
    """
    Main conversation engine for the capstone project that integrates
    all conversational components into a cohesive system.
    """
    
    def __init__(self, nim_endpoint: str = "http://localhost:8000"):
        # Initialize all conversational components
        self.speech_recognizer = ASRManager()
        self.dialogue_manager = DialogueManager()
        self.nlg_engine = NLGEngine()
        self.safety_manager = SafetyManager()
        self.llm_interface = None  # Will initialize when needed
        
        # Initialize NIM client and LLM interface if endpoint is available
        try:
            nim_client = NIMClient(nim_endpoint=nim_endpoint)
            self.llm_interface = ROS2LLMInterface(nim_client)
        except Exception:
            # LLM interface not available, continue with rule-based system
            pass
        
        self.status = ConversationStatus.IDLE
        self.current_context: Optional[ConversationContext] = None
        self.conversation_history: List[ConversationResult] = []
        self.is_active = False
        self.response_callbacks: List[Any] = []
    
    def start_conversation(self, user_id: str, mode: InteractionMode = InteractionMode.STANDALONE) -> bool:
        """Start a new conversation session"""
        session_id = f"session_{int(datetime.now().timestamp() * 1000)}"
        
        self.current_context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            current_mode=mode,
            conversation_history=[],
            user_profile={},
            task_context=None,
            last_interaction_time=datetime.now(),
            interruption_level=0
        )
        
        self.dialogue_manager.start_new_session(user_id)
        self.status = ConversationStatus.LISTENING
        self.is_active = True
        
        return True
    
    def process_text_input(self, text: str) -> ConversationResult:
        """Process text input directly (for testing or alternative input)"""
        start_time = datetime.now()
        
        if not self.is_active:
            return ConversationResult(
                success=False,
                user_input=text,
                system_response="",
                intent=None,
                entities={},
                execution_time=0.0,
                error_message="Conversation is not active"
            )
        
        # Apply safety filter to user input
        safety_result = self.safety_manager.check_input(text, self.current_context.user_id if self.current_context else None)
        if not safety_result.is_safe:
            # Filter the input or return a safe response
            filtered_text = safety_result.filtered_text if safety_result.filtered_text != text else "inappropriate content"
        else:
            filtered_text = text
        
        # Process through dialogue manager
        dialogue_response = self.dialogue_manager.process_input(filtered_text)
        
        # Generate response using NLG
        if self.current_context:
            response_text = self.nlg_engine.generate_contextual_response(
                dialogue_response.intent.value if hasattr(dialogue_response.intent, 'value') else str(dialogue_response.intent),
                dialogue_response.entities,
                {"is_continuation": True}
            )
        else:
            response_text = self.nlg_engine.generate_response(
                dialogue_response.intent.value if hasattr(dialogue_response.intent, 'value') else str(dialogue_response.intent),
                dialogue_response.entities
            )
        
        # Apply safety filter to response
        if self.current_context:
            filtered_response, safety_result = self.safety_manager.moderate_response(
                response_text, text, self.current_context.user_id
            )
        else:
            filtered_response, safety_result = self.safety_manager.moderate_response(
                response_text, text
            )
        
        # Update conversation history
        if self.current_context:
            self.current_context.conversation_history.append({
                "user": text,
                "system": filtered_response,
                "timestamp": datetime.now(),
                "intent": dialogue_response.intent.value if hasattr(dialogue_response.intent, 'value') else str(dialogue_response.intent)
            })
            self.current_context.last_interaction_time = datetime.now()
        
        # Create result
        result = ConversationResult(
            success=True,
            user_input=text,
            system_response=filtered_response,
            intent=dialogue_response.intent,
            entities=dialogue_response.entities,
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        self.conversation_history.append(result)
        
        # Trigger callbacks
        for callback in self.response_callbacks:
            try:
                callback(result)
            except Exception:
                pass  # Continue even if callback fails
        
        return result
    
    def process_speech_input(self, audio_data: Any) -> ConversationResult:
        """Process speech input through the full pipeline"""
        start_time = datetime.now()
        
        # Recognize speech
        recognition_result = self.speech_recognizer.recognize(audio_data)
        
        if not recognition_result.is_success or recognition_result.confidence < 0.5:
            # Low confidence recognition
            result = ConversationResult(
                success=False,
                user_input="",
                system_response="I'm sorry, I didn't catch that. Could you repeat?",
                intent=IntentType.UNDEFINED,
                entities={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message="Low confidence speech recognition"
            )
            self.conversation_history.append(result)
            return result
        
        # Process the recognized text
        return self.process_text_input(recognition_result.transcript)
    
    def get_system_response(self, user_input: str) -> str:
        """Get a response from the system based on user input"""
        # This would use the full conversational pipeline
        result = self.process_text_input(user_input)
        return result.system_response
    
    def update_user_profile(self, attributes: Dict[str, Any]):
        """Update the profile for the current user"""
        if self.current_context:
            self.current_context.user_profile.update(attributes)
            self.safety_manager.update_user_profile(self.current_context.user_id, attributes)
    
    def change_interaction_mode(self, mode: InteractionMode):
        """Change the current interaction mode"""
        if self.current_context:
            self.current_context.current_mode = mode
    
    def set_task_context(self, task_data: Dict[str, Any]):
        """Set context related to a specific task"""
        if self.current_context:
            self.current_context.task_context = task_data
    
    def get_conversation_status(self) -> ConversationStatus:
        """Get the current conversation status"""
        return self.status
    
    def is_conversation_active(self) -> bool:
        """Check if a conversation is currently active"""
        return self.is_active
    
    def end_conversation(self):
        """End the current conversation"""
        self.is_active = False
        self.status = ConversationStatus.IDLE
        self.current_context = None
    
    def add_response_callback(self, callback: Any):
        """Add a callback function to be called when a response is generated"""
        self.response_callbacks.append(callback)
    
    def get_conversation_history(self) -> List[ConversationResult]:
        """Get the history of conversation turns"""
        return self.conversation_history[:]
    
    def get_recent_context(self) -> Optional[ConversationContext]:
        """Get the current conversation context"""
        return self.current_context


class MultimodalConversationEngine(CapstoneConversationEngine):
    """
    Extended conversation engine that handles multimodal inputs 
    (speech, gesture, and other modalities)
    """
    
    def __init__(self, nim_endpoint: str = "http://localhost:8000"):
        super().__init__(nim_endpoint)
        self.gesture_interpretation_cache = {}
    
    def process_multimodal_input(self, speech: Optional[str] = None, 
                                gesture: Optional[Any] = None,
                                context: Optional[Dict[str, Any]] = None) -> ConversationResult:
        """Process input from multiple modalities"""
        combined_input = ""
        
        # Incorporate speech
        if speech:
            combined_input += speech
        
        # Incorporate gesture context if available
        if gesture:
            gesture_context = self._interpret_gesture(gesture)
            if gesture_context:
                combined_input += f" [Gesture: {gesture_context}]"
        
        # Process the combined input
        return self.process_text_input(combined_input)
    
    def _interpret_gesture(self, gesture_data: Any) -> Optional[str]:
        """Interpret gesture data into a textual description"""
        # This would connect to the perception system to interpret gestures
        # For now, return a mock interpretation
        # In a real system, this would analyze gesture data from the perception stack
        if hasattr(gesture_data, 'gesture_type'):
            return gesture_data.gesture_type.value
        return None


class ConversationTaskExecutor:
    """
    Executes tasks that come up during conversations
    """
    
    def __init__(self):
        self.active_tasks = {}
        self.task_queue = []
        self.is_running = False
    
    def register_task_handler(self, intent: IntentType, handler_func):
        """Register a function to handle a specific intent"""
        # In a real system, this would register the handler
        pass
    
    def execute_conversation_task(self, intent: IntentType, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on conversational intent"""
        # This would dispatch to appropriate task handlers based on intent
        # For example, navigation tasks, manipulation tasks, etc.
        
        if intent == IntentType.COMMAND:
            # Handle robot commands
            return self._execute_robot_command(entities)
        elif intent == IntentType.INFORMATION_REQUEST:
            # Handle information requests
            return self._execute_information_request(entities)
        else:
            # Default to no action
            return {"success": True, "action": "none", "message": "Task processed"}
    
    def _execute_robot_command(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a robot command from conversation"""
        # Extract action from entities
        actions = entities.get("actions", [])
        
        if actions:
            # Execute the first action
            action = actions[0]
            return {
                "success": True,
                "action": action,
                "message": f"Executing: {action}"
            }
        
        return {
            "success": False,
            "action": "none", 
            "message": "No specific action identified in command"
        }
    
    def _execute_information_request(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an information request from conversation"""
        # Extract request type
        request_types = entities.get("information_types", [])
        
        if request_types:
            req_type = request_types[0]
            return {
                "success": True,
                "request_type": req_type,
                "message": f"Looking up information about {req_type}"
            }
        
        return {
            "success": False,
            "request_type": "unknown",
            "message": "Could not identify information request type"
        }


class ConversationalSystemManager:
    """
    Main manager for the conversational system that coordinates all components
    """
    
    def __init__(self, nim_endpoint: str = "http://localhost:8000"):
        self.conversation_engine = MultimodalConversationEngine(nim_endpoint)
        self.task_executor = ConversationTaskExecutor()
        self.is_active = False
    
    def start_conversation_system(self, user_id: str, mode: InteractionMode = InteractionMode.STANDALONE):
        """Start the conversational system"""
        success = self.conversation_engine.start_conversation(user_id, mode)
        self.is_active = success
        return success
    
    def process_input(self, input_type: str, input_data: Any) -> ConversationResult:
        """Process input of various types"""
        if not self.is_active:
            return ConversationResult(
                success=False,
                user_input="",
                system_response="Conversation system is not active",
                intent=None,
                entities={},
                execution_time=0.0,
                error_message="System not active"
            )
        
        if input_type == "text":
            return self.conversation_engine.process_text_input(input_data)
        elif input_type == "speech":
            return self.conversation_engine.process_speech_input(input_data)
        elif input_type == "multimodal":
            speech, gesture = input_data
            return self.conversation_engine.process_multimodal_input(speech, gesture)
        else:
            return ConversationResult(
                success=False,
                user_input=str(input_data),
                system_response="Unsupported input type",
                intent=IntentType.UNDEFINED,
                entities={},
                execution_time=0.0,
                error_message=f"Unknown input type: {input_type}"
            )
    
    def get_response(self, user_input: str) -> str:
        """Get a response to user input"""
        result = self.process_input("text", user_input)
        return result.system_response
    
    def execute_conversation_task(self, intent: IntentType, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task that emerged from conversation"""
        return self.task_executor.execute_conversation_task(intent, entities)
    
    def update_user_context(self, user_profile: Dict[str, Any]):
        """Update the context for the current user"""
        self.conversation_engine.update_user_profile(user_profile)
    
    def set_task_context(self, task_data: Dict[str, Any]):
        """Set context related to a specific task"""
        self.conversation_engine.set_task_context(task_data)
    
    def is_system_active(self) -> bool:
        """Check if the conversational system is active"""
        return self.is_active
    
    def stop_conversation_system(self):
        """Stop the conversational system"""
        self.conversation_engine.end_conversation()
        self.is_active = False
    
    def get_conversation_history(self) -> List[ConversationResult]:
        """Get the conversation history"""
        return self.conversation_engine.get_conversation_history()
    
    def add_response_callback(self, callback: Any):
        """Add a callback for response events"""
        self.conversation_engine.add_response_callback(callback)