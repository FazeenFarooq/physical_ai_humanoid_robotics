"""
Conversational safety filters for the Physical AI & Humanoid Robotics course.
This module implements safety checks and content filtering for conversational systems.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from functools import wraps


class SafetyLevel(Enum):
    """Safety classification levels"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    DANGEROUS = "dangerous"


class SafetyCategory(Enum):
    """Categories of safety concerns"""
    OFFENSIVE_LANGUAGE = "offensive_language"
    PRIVACY_VIOLATION = "privacy_violation"
    PHYSICAL_HARM = "physical_harm"
    MISINFORMATION = "misinformation"
    SOCIAL_ENGINEERING = "social_engineering"
    UNSAFE_INSTRUCTIONS = "unsafe_instructions"
    DISCRIMINATION = "discrimination"
    INAPPROPRIATE_REQUEST = "inappropriate_request"


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    is_safe: bool
    safety_level: SafetyLevel
    categories: List[SafetyCategory]
    filtered_text: str
    severity_score: float  # 0.0 to 1.0
    explanation: str


class ContentFilter:
    """Base class for content filtering"""
    
    def __init__(self):
        self.offensive_keywords = set([
            # Explicit language
            "damn", "hell", "crap", "darn", "shoot",
            # Potentially offensive terms
            "stupid", "dumb", "idiot", "moron", "fool",
            # Potentially harmful instructions
            "hurt", "harm", "kill", "destroy", "break", "damage"
        ])
        
        # Regular expressions for pattern matching
        self.patterns = {
            SafetyCategory.OFFENSIVE_LANGUAGE: [
                r"\b(damn|hell|crap|darn|shoot)\b",
                r"\b(stupid|dumb|idiot|moron|fool)\b"
            ],
            SafetyCategory.PHYSICAL_HARM: [
                r"\b(hurt|harm|kill|destroy|break|damage|injure|attack|violence)\b",
                r"(how to )?(make|create|build).*(bomb|explosive|weapon)",
                r"(how to )?(steal|rob|burglarize|trespass)"
            ],
            SafetyCategory.PRIVACY_VIOLATION: [
                r"(get|find|obtain|steal).*(ssn|social security|credit card|password|pin)",
                r"(tell me|reveal).*(private|personal|confidential).*(information|info)"
            ],
            SafetyCategory.UNSAFE_INSTRUCTIONS: [
                r"(how to )?(jailbreak|root|hack).*(phone|device|system)",
                r"(how to )?(bypass|circumvent|evade).*security",
                r"(teach me|show me).*(unsafe|dangerous).*(activity|behavior|practice)"
            ],
            SafetyCategory.DISCRIMINATION: [
                r"\b(racist|sexist|homophobic|discriminatory|prejudice)\b",
                r"\b(hate|hatred).*(race|gender|religion|orientation)"
            ],
            SafetyCategory.SOCIAL_ENGINEERING: [
                r"(pretend to be|impersonate|pose as).*(authority|official|doctor|law enforcement)",
                r"(convince|trick|manipulate).*(person|people).*(into|to).*(sharing|giving|revealing)"
            ]
        }
    
    def check_text(self, text: str) -> SafetyCheckResult:
        """Check text for safety concerns"""
        text_lower = text.lower()
        detected_categories = set()
        severity_score = 0.0
        
        # Check for offensive keywords
        offensive_count = 0
        for keyword in self.offensive_keywords:
            if keyword in text_lower:
                offensive_count += 1
                if SafetyCategory.OFFENSIVE_LANGUAGE not in detected_categories:
                    detected_categories.add(SafetyCategory.OFFENSIVE_LANGUAGE)
                    severity_score = max(severity_score, 0.2)
        
        # Check against regex patterns
        for category, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_categories.add(category)
                    # Assign severity based on category
                    if category in [SafetyCategory.PHYSICAL_HARM, SafetyCategory.UNSAFE_INSTRUCTIONS]:
                        severity_score = max(severity_score, 0.9)
                    elif category in [SafetyCategory.PRIVACY_VIOLATION, SafetyCategory.SOCIAL_ENGINEERING]:
                        severity_score = max(severity_score, 0.8)
                    elif category in [SafetyCategory.DISCRIMINATION]:
                        severity_score = max(severity_score, 0.7)
                    else:
                        severity_score = max(severity_score, 0.5)
        
        # Determine safety level based on severity
        if severity_score >= 0.9:
            safety_level = SafetyLevel.DANGEROUS
        elif severity_score >= 0.7:
            safety_level = SafetyLevel.HIGH_RISK
        elif severity_score >= 0.5:
            safety_level = SafetyLevel.MEDIUM_RISK
        elif severity_score >= 0.3:
            safety_level = SafetyLevel.LOW_RISK
        else:
            safety_level = SafetyLevel.SAFE
        
        # Generate explanation
        if detected_categories:
            explanation = f"Detected categories: {', '.join([cat.value for cat in detected_categories])}"
        else:
            explanation = "No safety concerns detected"
        
        # Filter text by replacing offensive content with asterisks
        filtered_text = self._filter_offensive_content(text)
        
        return SafetyCheckResult(
            is_safe=(safety_level in [SafetyLevel.SAFE, SafetyLevel.LOW_RISK]),
            safety_level=safety_level,
            categories=list(detected_categories),
            filtered_text=filtered_text,
            severity_score=severity_score,
            explanation=explanation
        )
    
    def _filter_offensive_content(self, text: str) -> str:
        """Filter offensive content in text"""
        filtered = text
        for keyword in self.offensive_keywords:
            # Replace with asterisks of same length
            filtered = re.sub(
                r'\b' + re.escape(keyword) + r'\b',
                '*' * len(keyword),
                filtered,
                flags=re.IGNORECASE
            )
        return filtered


class ContextAwareSafetyFilter(ContentFilter):
    """Advanced safety filter that considers conversation context"""
    
    def __init__(self):
        super().__init__()
        self.conversation_context = []
        self.user_profiles = {}
        self.safety_threshold = 0.5  # Threshold above which responses are filtered
    
    def add_context(self, user_id: str, message: str, timestamp: float = None):
        """Add a message to the conversation context"""
        if timestamp is None:
            timestamp = time.time()
        
        context_entry = {
            "user_id": user_id,
            "message": message,
            "timestamp": timestamp,
            "safety_check": None  # Will be populated after safety check
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only recent context (last 10 messages)
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def check_with_context(self, text: str, user_id: str = None) -> SafetyCheckResult:
        """Check text with additional context considerations"""
        # First, perform basic content check
        base_result = self.check_text(text)
        
        # If user ID is provided, consider user-specific factors
        if user_id and user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            # Adjust severity based on user trust level, history, etc.
            if user_profile.get("trust_level", "neutral") == "high":
                base_result.severity_score = max(0.0, base_result.severity_score - 0.2)
        
        # Consider conversation context
        if self.conversation_context:
            # Check if this continues a potentially unsafe conversation
            recent_messages = [
                entry for entry in self.conversation_context[-3:] 
                if entry.get("user_id") == user_id or user_id is None
            ]
            
            # If recent messages have safety concerns, increase severity of current message
            for msg in recent_messages:
                if msg.get("safety_check") and msg["safety_check"].severity_score > 0.5:
                    base_result.severity_score = min(1.0, base_result.severity_score + 0.1)
        
        # Update safety level based on adjusted severity
        if base_result.severity_score >= 0.9:
            base_result.safety_level = SafetyLevel.DANGEROUS
        elif base_result.severity_score >= 0.7:
            base_result.safety_level = SafetyLevel.HIGH_RISK
        elif base_result.severity_score >= 0.5:
            base_result.safety_level = SafetyLevel.MEDIUM_RISK
        elif base_result.severity_score >= 0.3:
            base_result.safety_level = SafetyLevel.LOW_RISK
        else:
            base_result.safety_level = SafetyLevel.SAFE
        
        # Update is_safe flag based on threshold
        base_result.is_safe = base_result.safety_level in [SafetyLevel.SAFE, SafetyLevel.LOW_RISK]
        
        return base_result
    
    def moderate_response(self, response: str, request: str, user_id: str = None) -> Tuple[str, SafetyCheckResult]:
        """Moderate a response based on the original request and user context"""
        # Check the response for safety
        response_result = self.check_with_context(response, user_id)
        
        if not response_result.is_safe and response_result.severity_score >= self.safety_threshold:
            # If response is unsafe, provide a safe alternative
            safe_response = self._generate_safe_alternative(request, response_result)
            return safe_response, response_result
        else:
            # Response is safe, but we might want to filter offensive content
            filtered_response = self._filter_offensive_content(response)
            return filtered_response, response_result
    
    def _generate_safe_alternative(self, request: str, safety_result: SafetyCheckResult) -> str:
        """Generate a safe alternative response when the original is flagged"""
        # Determine appropriate response based on detected categories
        if SafetyCategory.PHYSICAL_HARM in safety_result.categories:
            return "I can't provide information about causing harm to people or property. Is there something positive I can help you with instead?"
        elif SafetyCategory.PRIVACY_VIOLATION in safety_result.categories:
            return "I can't provide information about obtaining private information. Protecting privacy is important."
        elif SafetyCategory.UNSAFE_INSTRUCTIONS in safety_result.categories:
            return "I can't provide instructions for potentially dangerous activities. I'm here to help with safe, positive interactions."
        elif SafetyCategory.DISCRIMINATION in safety_result.categories:
            return "I don't engage with discriminatory content. I'm designed to treat all users with respect and equality."
        else:
            return "I'm not able to respond to that in a helpful way. Could you ask about something more positive and constructive?"


class RobotSafetyFilter(ContextAwareSafetyFilter):
    """Safety filter specifically tailored for robotic applications"""
    
    def __init__(self):
        super().__init__()
        # Add robot-specific patterns
        self.robot_specific_patterns = {
            SafetyCategory.UNSAFE_INSTRUCTIONS: [
                r"(make|let|allow).*(robot|me|you).*(harm|injure|damage|destroy).*(person|people|human|object)",
                r"(tell|instruct|program).*(robot|you).*(to ).*(attack|violence|destroy|damage|harm)",
                r"(how to )?(disable|bypass|override).*(safety|emergency|stop).*(feature|protocol|system)",
                r"(ignore|disregard).*(safety|ethical|protocol).*(guidelines|rules|measures)"
            ],
            SafetyCategory.PHYSICAL_HARM: [
                r"(move to|go to|navigate to).*(dangerous|unsafe|high.*risk|construction.*zone|restricted.*area)",
                r"(pick up|grasp|handle).*(sharp|breakable|dangerous|fragile|hot).*(object|item|thing)"
            ]
        }
    
    def check_robot_command(self, command: str, user_id: str = None) -> SafetyCheckResult:
        """Check a robot command for safety"""
        # First, perform standard content check
        result = self.check_with_context(command, user_id)
        
        # Then, check against robot-specific patterns
        command_lower = command.lower()
        for category, pattern_list in self.robot_specific_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, command_lower, re.IGNORECASE):
                    if category not in result.categories:
                        result.categories.append(category)
                        
                        # Update severity based on robot-specific concerns
                        if category == SafetyCategory.UNSAFE_INSTRUCTIONS:
                            result.severity_score = max(result.severity_score, 0.9)
                        elif category == SafetyCategory.PHYSICAL_HARM:
                            result.severity_score = max(result.severity_score, 0.8)
        
        # Update safety level and is_safe flag based on new severity
        if result.severity_score >= 0.9:
            result.safety_level = SafetyLevel.DANGEROUS
        elif result.severity_score >= 0.7:
            result.safety_level = SafetyLevel.HIGH_RISK
        elif result.severity_score >= 0.5:
            result.safety_level = SafetyLevel.MEDIUM_RISK
        elif result.severity_score >= 0.3:
            result.safety_level = SafetyLevel.LOW_RISK
        else:
            result.safety_level = SafetyLevel.SAFE
        
        result.is_safe = result.safety_level in [SafetyLevel.SAFE, SafetyLevel.LOW_RISK]
        
        return result
    
    def check_robot_response(self, response: str, user_request: str, user_id: str = None) -> Tuple[str, SafetyCheckResult]:
        """Check and potentially modify a robot's response"""
        # First check if the response is safe
        response_result = self.check_with_context(response, user_id)
        
        # If the response contains unsafe commands for the robot
        if any(cat in [SafetyCategory.UNSAFE_INSTRUCTIONS, SafetyCategory.PHYSICAL_HARM] 
               for cat in response_result.categories):
            # Generate a safe alternative
            safe_response = "I cannot perform actions that might cause harm. I'm designed to operate safely around people and property. Is there a safe task I can help with instead?"
            return safe_response, response_result
        elif not response_result.is_safe and response_result.severity_score >= self.safety_threshold:
            # For other safety concerns, generate appropriate response
            safe_response = self._generate_safe_alternative(user_request, response_result)
            return safe_response, response_result
        else:
            # Response is safe, return filtered version
            filtered_response = self._filter_offensive_content(response)
            return filtered_response, response_result


class SafetyManager:
    """
    Main safety management system that coordinates all safety checks
    for conversational robotics applications.
    """
    
    def __init__(self):
        self.robot_filter = RobotSafetyFilter()
        self.enabled = True
        self.log_entries = []
        self.max_log_entries = 1000
    
    def enable_filtering(self):
        """Enable safety filtering"""
        self.enabled = True
    
    def disable_filtering(self):
        """Disable safety filtering (use with caution)"""
        self.enabled = False
    
    def check_input(self, text: str, user_id: str = None) -> SafetyCheckResult:
        """Check user input for safety concerns"""
        if not self.enabled:
            return SafetyCheckResult(
                is_safe=True,
                safety_level=SafetyLevel.SAFE,
                categories=[],
                filtered_text=text,
                severity_score=0.0,
                explanation="Safety filtering disabled"
            )
        
        result = self.robot_filter.check_with_context(text, user_id)
        self._log_check("input", text, result)
        return result
    
    def check_robot_command(self, command: str, user_id: str = None) -> SafetyCheckResult:
        """Check a robot command for safety"""
        if not self.enabled:
            return SafetyCheckResult(
                is_safe=True,
                safety_level=SafetyLevel.SAFE,
                categories=[],
                filtered_text=command,
                severity_score=0.0,
                explanation="Safety filtering disabled"
            )
        
        result = self.robot_filter.check_robot_command(command, user_id)
        self._log_check("robot_command", command, result)
        return result
    
    def moderate_response(self, response: str, request: str, user_id: str = None) -> Tuple[str, SafetyCheckResult]:
        """Moderate a response before sending to user"""
        if not self.enabled:
            return response, SafetyCheckResult(
                is_safe=True,
                safety_level=SafetyLevel.SAFE,
                categories=[],
                filtered_text=response,
                severity_score=0.0,
                explanation="Safety filtering disabled"
            )
        
        filtered_response, result = self.robot_filter.check_robot_response(response, request, user_id)
        self._log_check("response", response, result, additional_info={"original_response": response})
        return filtered_response, result
    
    def add_user_context(self, user_id: str, message: str):
        """Add user message to context for contextual safety checking"""
        self.robot_filter.add_context(user_id, message)
    
    def _log_check(self, check_type: str, original_text: str, result: SafetyCheckResult, 
                   additional_info: Dict[str, Any] = None):
        """Log a safety check for auditing and analysis"""
        log_entry = {
            "timestamp": time.time(),
            "type": check_type,
            "original_text": original_text[:100],  # Truncate for privacy
            "safety_level": result.safety_level.value,
            "categories": [cat.value for cat in result.categories],
            "severity_score": result.severity_score,
            "is_safe": result.is_safe,
            "additional_info": additional_info or {}
        }
        
        self.log_entries.append(log_entry)
        
        # Keep only recent logs to manage memory
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get a report of recent safety checks"""
        total_checks = len(self.log_entries)
        if total_checks == 0:
            return {"message": "No safety checks performed yet"}
        
        unsafe_count = sum(1 for entry in self.log_entries 
                          if entry["safety_level"] in ["medium_risk", "high_risk", "dangerous"])
        
        category_counts = {}
        for entry in self.log_entries:
            for cat in entry["categories"]:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_checks": total_checks,
            "unsafe_count": unsafe_count,
            "unsafe_percentage": (unsafe_count / total_checks) * 100,
            "most_common_issues": sorted(category_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:5],
            "recent_logs": self.log_entries[-10:]  # Last 10 entries
        }
    
    def update_user_profile(self, user_id: str, attributes: Dict[str, Any]):
        """Update user profile for personalized safety filtering"""
        if user_id not in self.robot_filter.user_profiles:
            self.robot_filter.user_profiles[user_id] = {}
        self.robot_filter.user_profiles[user_id].update(attributes)


# Decorator for easy safety checking
def safety_check(filter_manager: SafetyManager, user_id_param: str = "user_id"):
    """Decorator to automatically check safety of function inputs"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args
            user_id = kwargs.get(user_id_param)
            if user_id is None:
                # Try to find user_id in args if it's the first parameter
                if args and isinstance(args[0], str):
                    user_id = args[0]
            
            # Get the text to check (usually the first argument after user_id)
            text_to_check = None
            if args and isinstance(args[-1], str):
                text_to_check = args[-1]
            
            if text_to_check:
                # Perform safety check
                result = filter_manager.check_input(text_to_check, user_id)
                
                if not result.is_safe:
                    # If the content is unsafe, return a safe response
                    return f"Content flagged as unsafe: {result.explanation}"
            
            # If content is safe, proceed with original function
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        return wrapper
    return decorator


# Utility function for batch safety checking
def batch_safety_check(texts: List[str], safety_manager: SafetyManager, 
                      user_id: str = None) -> List[SafetyCheckResult]:
    """Check safety for multiple texts at once"""
    results = []
    for text in texts:
        result = safety_manager.check_input(text, user_id)
        results.append(result)
    return results