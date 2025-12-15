"""
Dialogue management system for the Physical AI & Humanoid Robotics course.
This module manages conversational flow and context in human-robot interaction.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import enum
import time
import re
from datetime import datetime


class DialogueState(enum.Enum):
    """States in the dialogue management system"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_INPUT = "waiting_for_input"
    ERROR = "error"


class IntentType(enum.Enum):
    """Types of recognized intents"""
    GREETING = "greeting"
    COMMAND = "command"
    INFORMATION_REQUEST = "information_request"
    CHITCHAT = "chitchat"
    GOODBYE = "goodbye"
    UNDEFINED = "undefined"


@dataclass
class DialogueContext:
    """Context information for current dialogue"""
    session_id: str
    user_id: Optional[str]
    timestamp: datetime
    turn_count: int
    previous_intent: Optional[IntentType]
    current_intent: Optional[IntentType]
    entities: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    user_profile: Dict[str, Any]


@dataclass
class DialogueAct:
    """Represents an action in the dialogue"""
    intent: IntentType
    entities: Dict[str, Any]
    confidence: float
    response_template: str
    required_context: List[str]


@dataclass
class DialogueResponse:
    """Response from the dialogue system"""
    text: str
    intent: IntentType
    entities: Dict[str, Any]
    dialogue_state: DialogueState
    next_expected_input: Optional[str] = None
    actions: List[str] = None


class IntentClassifier:
    """Classifies user input into intents"""
    
    def __init__(self):
        # Define patterns for different intents
        self.patterns = {
            IntentType.GREETING: [
                r"hello", r"hi", r"hey", r"greetings", r"good morning", 
                r"good afternoon", r"good evening"
            ],
            IntentType.COMMAND: [
                r"go to", r"move to", r"pick up", r"grasp", r"stop", 
                r"turn", r"go forward", r"go back", r"take", r"bring"
            ],
            IntentType.INFORMATION_REQUEST: [
                r"what", r"how", r"where", r"when", r"who", 
                "time", r"temperature", r"status", r"location"
            ],
            IntentType.CHITCHAT: [
                r"how are you", r"what's up", r"tell me", r"joke", 
                r"story", r"fun fact", r"random"
            ],
            IntentType.GOODBYE: [
                r"bye", r"goodbye", r"see you", r"farewell", 
                r"talk to you later", r"exit"
            ]
        }
    
    def classify(self, text: str) -> tuple[IntentType, float]:
        """Classify text into an intent with confidence score"""
        text_lower = text.lower()
        max_score = 0.0
        best_intent = IntentType.UNDEFINED
        
        for intent, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(r"\b" + pattern + r"\b", text_lower):
                    score += 1
            
            # Normalize score by number of patterns
            score = score / len(patterns)
            if score > max_score:
                max_score = score
                best_intent = intent
        
        # Convert to confidence (0.0 to 1.0)
        confidence = min(1.0, max_score * 2)  # Boost score slightly
        
        return best_intent, confidence


class EntityExtractor:
    """Extracts named entities from user input"""
    
    def __init__(self):
        self.location_entities = [
            "kitchen", "living room", "bedroom", "office", "bathroom", 
            "dining room", "hallway", "garage", "garden", "entrance"
        ]
        
        self.object_entities = [
            "ball", "book", "cup", "bottle", "phone", "keys", 
            "toy", "box", "chair", "table", "red object", "blue object"
        ]
        
        self.action_entities = [
            "pick up", "grasp", "move to", "go to", "bring", 
            "take", "place", "put down", "lift", "lower"
        ]
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {
            "locations": [],
            "objects": [],
            "actions": [],
            "quantities": [],
            "times": []
        }
        
        text_lower = text.lower()
        
        # Extract location entities
        for loc in self.location_entities:
            if loc in text_lower:
                entities["locations"].append(loc)
        
        # Extract object entities
        for obj in self.object_entities:
            if obj in text_lower:
                entities["objects"].append(obj)
        
        # Extract action entities
        for act in self.action_entities:
            if act in text_lower:
                entities["actions"].append(act)
        
        # Extract quantities (simple number extraction)
        quantity_matches = re.findall(r"\d+", text)
        entities["quantities"] = [int(q) for q in quantity_matches]
        
        return entities


class DialoguePolicy:
    """Determines the next action in the dialogue based on context"""
    
    def __init__(self):
        self.responses = {
            IntentType.GREETING: [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I help you?"
            ],
            IntentType.COMMAND: [
                "I will execute that command now.",
                "Processing your request...",
                "Got it, performing the action."
            ],
            IntentType.INFORMATION_REQUEST: [
                "I'm looking up that information for you.",
                "Let me check that for you.",
                "I can help with that question."
            ],
            IntentType.CHITCHAT: [
                "That's an interesting topic!",
                "I'd be happy to chat about that.",
                "Tell me more about that."
            ],
            IntentType.GOODBYE: [
                "Goodbye! Have a great day!",
                "See you later!",
                "Farewell! It was nice talking to you."
            ]
        }
    
    def get_response(self, intent: IntentType, entities: Dict[str, Any] = None) -> str:
        """Get an appropriate response for the intent"""
        if entities is None:
            entities = {}
        
        responses = self.responses.get(intent, ["I'm not sure how to respond to that."])
        
        # Select a response (in a more sophisticated system, this could be learned)
        import random
        response_template = random.choice(responses)
        
        # For now, return the template; in a real system, you'd fill in variables
        return response_template


class DialogueManager:
    """
    Main dialogue management system that orchestrates conversation flow.
    Based on the requirements in the data model and API contracts.
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.dialogue_policy = DialoguePolicy()
        self.current_context: Optional[DialogueContext] = None
        self.session_history: Dict[str, List[DialogueContext]] = {}
        self.state = DialogueState.IDLE
        self.response_callbacks: List[Callable[[DialogueResponse], None]] = []
    
    def start_new_session(self, user_id: Optional[str] = None) -> str:
        """Start a new dialogue session"""
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.current_context = DialogueContext(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            turn_count=0,
            previous_intent=None,
            current_intent=None,
            entities={},
            conversation_history=[],
            user_profile={}
        )
        
        self.state = DialogueState.LISTENING
        return session_id
    
    def process_input(self, user_input: str) -> DialogueResponse:
        """Process user input and generate an appropriate response"""
        if self.current_context is None:
            self.start_new_session()
        
        # Update turn count
        self.current_context.turn_count += 1
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify(user_input)
        
        # Extract entities
        entities = self.entity_extractor.extract(user_input)
        
        # Update context
        self.current_context.previous_intent = self.current_context.current_intent
        self.current_context.current_intent = intent
        self.current_context.entities = entities
        
        # Add to conversation history
        self.current_context.conversation_history.append({
            "turn": self.current_context.turn_count,
            "speaker": "user",
            "text": user_input,
            "intent": intent.value,
            "entities": entities,
            "timestamp": datetime.now()
        })
        
        # Generate response
        response_text = self.dialogue_policy.get_response(intent, entities)
        
        # Update state based on intent
        if intent == IntentType.GOODBYE:
            self.state = DialogueState.IDLE
        else:
            self.state = DialogueState.RESPONDING
        
        # Create response
        response = DialogueResponse(
            text=response_text,
            intent=intent,
            entities=entities,
            dialogue_state=self.state,
            actions=self._determine_actions(intent, entities)
        )
        
        # Add to conversation history
        self.current_context.conversation_history.append({
            "turn": self.current_context.turn_count,
            "speaker": "system",
            "text": response_text,
            "intent": "system_response",
            "entities": {},
            "timestamp": datetime.now()
        })
        
        # Trigger callbacks
        for callback in self.response_callbacks:
            callback(response)
        
        return response
    
    def _determine_actions(self, intent: IntentType, entities: Dict[str, Any]) -> List[str]:
        """Determine what actions should follow the response"""
        actions = []
        
        if intent == IntentType.COMMAND:
            # Determine robot actions based on command
            if entities.get("locations"):
                actions.append(f"navigate_to:{entities['locations'][0]}")
            if entities.get("objects"):
                actions.append(f"pick_up:{entities['objects'][0]}")
            if entities.get("actions"):
                for action in entities["actions"]:
                    actions.append(f"execute:{action}")
        
        elif intent == IntentType.GOODBYE:
            actions.append("robot_shutdown")
        
        elif intent == IntentType.INFORMATION_REQUEST:
            # Determine what info is being requested
            text = " ".join([item for sublist in entities.values() for item in sublist])
            if "time" in text:
                actions.append("get_current_time")
            elif "status" in text:
                actions.append("get_system_status")
            elif "location" in text:
                actions.append("get_robot_location")
        
        return actions
    
    def add_response_callback(self, callback: Callable[[DialogueResponse], None]):
        """Add a callback to be called when a response is generated"""
        self.response_callbacks.append(callback)
    
    def get_current_context(self) -> Optional[DialogueContext]:
        """Get the current dialogue context"""
        return self.current_context
    
    def reset_session(self):
        """Reset the current session"""
        self.current_context = None
        self.state = DialogueState.IDLE
    
    def continue_conversation(self, user_input: str) -> DialogueResponse:
        """Continue an existing conversation"""
        if self.current_context is None:
            return self.process_input(user_input)
        
        # Process the input in the context of the current conversation
        return self.process_input(user_input)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a dialogue session"""
        if session_id in self.session_history:
            sessions = self.session_history[session_id]
            if sessions:
                latest_session = sessions[-1]  # Get the most recent session
                return {
                    "session_id": latest_session.session_id,
                    "turn_count": latest_session.turn_count,
                    "start_time": latest_session.timestamp,
                    "intents_exchanged": [item["intent"] for item in latest_session.conversation_history],
                    "entities_identified": latest_session.entities
                }
        
        return {}


class GoalOrientedDialogueManager(DialogueManager):
    """
    Extended dialogue manager for goal-oriented conversations
    where the robot needs to achieve specific tasks through dialogue
    """
    
    def __init__(self):
        super().__init__()
        self.active_goals: List[Dict[str, Any]] = []
        self.goal_context: Dict[str, Any] = {}
    
    def set_goal(self, goal_description: str, goal_parameters: Dict[str, Any] = None):
        """Set a goal for the dialogue system to achieve"""
        goal_id = f"goal_{int(time.time() * 1000)}"
        
        goal = {
            "id": goal_id,
            "description": goal_description,
            "parameters": goal_parameters or {},
            "status": "active",  # active, completed, failed
            "created_at": datetime.now()
        }
        
        self.active_goals.append(goal)
        return goal_id
    
    def update_goal_context(self, key: str, value: Any):
        """Update context relevant to goal achievement"""
        self.goal_context[key] = value
    
    def check_goal_completion(self) -> Optional[str]:
        """Check if any active goals have been completed"""
        for goal in self.active_goals:
            if goal["status"] == "completed":
                completed_id = goal["id"]
                self.active_goals.remove(goal)
                return completed_id
        
        return None