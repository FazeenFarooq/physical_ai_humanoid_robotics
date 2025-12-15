"""
Natural Language Generation (NLG) for the Physical AI & Humanoid Robotics course.
This module generates natural language responses for conversational robotics.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import random
import re
from datetime import datetime
import math


@dataclass
class NLGTemplate:
    """Template for generating natural language responses"""
    intent: str
    templates: List[str]
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 1


class NLGDatabase:
    """Database of response templates for different intents and contexts"""
    
    def __init__(self):
        self.templates = {
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I help you?",
                "Hello! It's great to see you."
            ],
            "goodbye": [
                "Goodbye! Have a wonderful day!",
                "See you later!",
                "Farewell! It was wonderful talking with you.",
                "Take care! Feel free to come back if you need anything."
            ],
            "command_acknowledged": [
                "I will execute that command now.",
                "Processing your request...",
                "Got it, performing the action.",
                "Understood, I'll carry out that task right away."
            ],
            "command_failed": [
                "I'm sorry, I couldn't complete that task.",
                "I encountered an issue while executing your command.",
                "I wasn't able to complete that action. Could you try rephrasing?",
                "I'm having trouble with that request. Would you like me to try something else?"
            ],
            "information_request": [
                "I'm looking up that information for you.",
                "Let me check that for you.",
                "I can help with that question.",
                "I'll find that information for you."
            ],
            "chitchat": [
                "That's an interesting topic!",
                "I'd be happy to chat about that.",
                "Tell me more about that.",
                "That's quite fascinating. What else would you like to discuss?"
            ],
            "navigation": [
                "I'm on my way to the {location}.",
                "Heading to the {location} now.",
                "Moving toward the {location}.",
                "I'll navigate to the {location} for you."
            ],
            "object_detection": [
                "I found the {object}.",
                "The {object} is right here.",
                "I can see the {object} in front of me.",
                "Located the {object} successfully."
            ],
            "error": [
                "I'm sorry, I didn't understand that.",
                "Could you please rephrase that?",
                "I'm not sure what you mean. Could you clarify?",
                "I encountered an issue processing your request."
            ]
        }
    
    def get_template(self, intent: str) -> List[str]:
        """Get templates for a specific intent"""
        return self.templates.get(intent, self.templates["error"])
    
    def add_template(self, intent: str, templates: List[str]):
        """Add templates for a specific intent"""
        if intent not in self.templates:
            self.templates[intent] = []
        self.templates[intent].extend(templates)


class NLGEngine:
    """
    Main Natural Language Generation engine that creates appropriate responses
    based on intent, context, and user profile.
    """
    
    def __init__(self):
        self.template_db = NLGDatabase()
        self.user_context = {}
        self.conversation_history = []
        self.system_info = {
            "robot_name": "AI Assistant",
            "current_time": datetime.now(),
            "location": "lab",
            "battery_level": 0.85
        }
    
    def generate_response(self, intent: str, entities: Dict[str, Any] = None, 
                         user_context: Dict[str, Any] = None) -> str:
        """Generate a natural language response based on intent and entities"""
        if entities is None:
            entities = {}
        if user_context is None:
            user_context = {}
        
        # Get response templates for the intent
        templates = self.template_db.get_template(intent)
        
        if not templates:
            templates = self.template_db.get_template("error")
        
        # Select a template (we'll use a simple random selection)
        # In a more sophisticated system, this selection could be contextual
        template = random.choice(templates)
        
        # Fill in any placeholders in the template with entities
        response = self._fill_placeholders(template, entities)
        
        # Apply user context and system context if needed
        response = self._apply_context(response, user_context)
        
        # Add to conversation history
        self.conversation_history.append({
            "intent": intent,
            "entities": entities,
            "response": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    def _fill_placeholders(self, template: str, entities: Dict[str, Any]) -> str:
        """Fill in placeholders in the template with entity values"""
        result = template
        
        # Replace placeholders like {location}, {object}, etc.
        # with actual values from entities
        for key, value in entities.items():
            if isinstance(value, list) and value:
                # Use the first item if it's a list
                result = result.replace(f"{{{key}}}", str(value[0]))
            else:
                result = result.replace(f"{{{key}}}", str(value))
        
        # Handle special system placeholders
        result = result.replace("{robot_name}", self.system_info["robot_name"])
        result = result.replace("{location}", self.system_info["location"])
        result = result.replace("{time}", self.system_info["current_time"].strftime("%H:%M"))
        
        return result
    
    def _apply_context(self, response: str, user_context: Dict[str, Any]) -> str:
        """Apply user context to personalize the response"""
        # Personalize based on known user information
        if "user_name" in user_context:
            response = response.replace("you", f"{user_context['user_name']}")
            # More sophisticated systems would replace more pronouns and use more complex personalization
        
        # Adjust formality based on context
        if user_context.get("formality_level") == "formal":
            # Replace casual phrases with more formal ones
            casual_to_formal = {
                "Hi": "Hello",
                "Hey": "Hello",
                "Cool": "Excellent",
                "Awesome": "Excellent"
            }
            for casual, formal in casual_to_formal.items():
                response = response.replace(casual, formal)
        
        return response
    
    def generate_multimodal_response(self, intent: str, entities: Dict[str, Any] = None,
                                   actions: List[str] = None) -> Dict[str, Any]:
        """Generate a response that includes both verbal and action components"""
        if entities is None:
            entities = {}
        if actions is None:
            actions = []
        
        # Generate verbal response
        verbal_response = self.generate_response(intent, entities)
        
        # Combine with appropriate actions
        multimodal_response = {
            "verbal_response": verbal_response,
            "actions": actions,
            "intent": intent,
            "entities": entities
        }
        
        return multimodal_response
    
    def generate_explanatory_response(self, action_result: Dict[str, Any]) -> str:
        """Generate a response that explains the result of an action"""
        success = action_result.get("success", False)
        action_type = action_result.get("action_type", "unknown")
        
        if success:
            if action_type == "navigation":
                dest = action_result.get("destination", "target location")
                return f"I have successfully reached the {dest}. How else can I assist you?"
            elif action_type == "manipulation":
                obj = action_result.get("object", "object")
                return f"I have successfully grasped the {obj}. What should I do with it?"
            elif action_type == "search":
                target = action_result.get("target", "object")
                location = action_result.get("location", "area")
                return f"I found the {target} in the {location}."
            else:
                return "I have completed the requested action successfully."
        else:
            error_msg = action_result.get("error", "an issue occurred")
            return f"I'm sorry, I encountered {error_msg} while performing that action."
    
    def generate_contextual_response(self, intent: str, entities: Dict[str, Any] = None,
                                   conversation_context: Dict[str, Any] = None) -> str:
        """Generate a response that takes into account the broader conversation context"""
        if entities is None:
            entities = {}
        if conversation_context is None:
            conversation_context = {}
        
        # Apply context to determine more appropriate response
        time_of_day = self._get_time_of_day()
        
        # Special handling for greetings based on time of day
        if intent == "greeting":
            templates = self.template_db.get_template("greeting")
            if time_of_day == "morning":
                # Prefer morning-specific greetings
                morning_greetings = [
                    f"Good morning! {resp.split('! ')[1] if '! ' in resp else resp}" 
                    for resp in templates
                ]
                template = random.choice(morning_greetings)
            elif time_of_day == "afternoon":
                template = random.choice(templates)
                if not template.lower().startswith("good afternoon"):
                    template = f"Good afternoon! {template}"
            elif time_of_day == "evening":
                template = random.choice(templates)
                if not template.lower().startswith("good evening"):
                    template = f"Good evening! {template}"
            else:
                template = random.choice(templates)
            
            return self._fill_placeholders(template, entities)
        
        # For subsequent utterances in a conversation
        if conversation_context.get("is_continuation", False):
            # Adjust response to flow naturally from previous conversation
            pass
        
        # Default behavior
        return self.generate_response(intent, entities)
    
    def _get_time_of_day(self) -> str:
        """Determine the time of day based on current time"""
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 21:
            return "evening"
        else:
            return "night"


class AdaptiveNLGEngine(NLGEngine):
    """
    Extended NLG engine that adapts its responses based on user interaction patterns
    and feedback over time.
    """
    
    def __init__(self):
        super().__init__()
        self.user_preferences = {}
        self.response_effectiveness = {}  # Track how effective different responses are
        self.personalized_templates = {}  # User-specific templates
    
    def learn_from_interaction(self, user_id: str, intent: str, response: str, 
                              user_feedback: str = None):
        """Learn from user interactions to improve future responses"""
        # This would track how well different responses work with different users
        # For now, we'll keep track of preferred response styles
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_style": "neutral",  # casual, formal, direct
                "response_length_preference": "medium",  # short, medium, long
                "interaction_history": []
            }
        
        interaction = {
            "intent": intent,
            "response": response,
            "feedback": user_feedback,
            "timestamp": datetime.now()
        }
        
        self.user_preferences[user_id]["interaction_history"].append(interaction)
    
    def generate_personalized_response(self, user_id: str, intent: str, 
                                    entities: Dict[str, Any] = None) -> str:
        """Generate a response personalized to the specific user"""
        if entities is None:
            entities = {}
        
        # Apply user preferences to response generation
        user_pref = self.user_preferences.get(user_id, {})
        
        # Get base response
        base_response = self.generate_response(intent, entities)
        
        # Apply personalization based on user preferences
        if user_pref.get("preferred_style") == "casual":
            # Make the response more casual
            formal_terms = {
                "Greetings": "Hey",
                "How may I help you": "What's up?",
                "It is my pleasure": "Happy to",
                "Certainly": "Sure"
            }
            for formal, casual in formal_terms.items():
                base_response = base_response.replace(formal, casual)
        
        elif user_pref.get("preferred_style") == "formal":
            # Make the response more formal
            casual_terms = {
                "Hi": "Hello",
                "Hey": "Hello",
                "Cool": "Appropriate",
                "Awesome": "Excellent"
            }
            for casual, formal in casual_terms.items():
                base_response = base_response.replace(casual, formal)
        
        # Adjust length based on preferences
        if user_pref.get("response_length_preference") == "short":
            # Truncate to first sentence
            sentences = base_response.split('.')
            if len(sentences) > 1:
                base_response = sentences[0] + '.'
        
        return base_response
    
    def get_dynamic_template(self, user_id: str, intent: str) -> Optional[str]:
        """Get a user-specific template if available"""
        user_templates = self.personalized_templates.get(user_id, {})
        return user_templates.get(intent)


# Utility function for response generation
def create_system_response(message: str, response_type: str = "info") -> Dict[str, Any]:
    """Create a standardized system response with metadata"""
    return {
        "message": message,
        "type": response_type,  # info, warning, error, success
        "timestamp": datetime.now().isoformat(),
        "source": "nlg_system"
    }