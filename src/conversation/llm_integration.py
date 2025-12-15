"""
LLM integration with ROS 2 system using NVIDIA NIM for the Physical AI & Humanoid Robotics course.
This module integrates large language models with the ROS 2 framework using NVIDIA NIM.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time
import threading
from enum import Enum


class LLMModelType(Enum):
    """Types of LLM models available through NIM"""
    GPT4 = "gpt-4"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    LLAMA2_7B = "llama2-7b"
    LLAMA2_13B = "llama2-13b"
    NIM_CUSTOM = "nim-custom"


@dataclass
class LLMRequest:
    """Request to the LLM service"""
    prompt: str
    model: LLMModelType
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    user_context: Dict[str, Any] = None


@dataclass
class LLMResponse:
    """Response from the LLM service"""
    text: str
    model: LLMModelType
    tokens_used: int
    processing_time: float
    confidence: float
    parsed_actions: List[Dict[str, Any]] = None  # Parsed robot actions


class NIMClient:
    """
    Client for interacting with NVIDIA NIM (NIM Inference Microservice)
    """
    
    def __init__(self, nim_endpoint: str = "http://localhost:8000", 
                 api_key: Optional[str] = None):
        self.nim_endpoint = nim_endpoint
        self.api_key = api_key
        self.session = requests.Session()
        
        # Add headers for authentication if API key is provided
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Default model configuration
        self.default_model = LLMModelType.LLAMA2_7B
        self.default_max_tokens = 512
        self.default_temperature = 0.7
    
    def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using the NIM service"""
        start_time = time.time()
        
        # Prepare the payload for the NIM service
        payload = {
            "prompt": request.prompt,
            "model": request.model.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
        
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        
        try:
            # Make the request to the NIM service
            response = self.session.post(
                f"{self.nim_endpoint}/generate",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract the generated text
            generated_text = result.get("text", result.get("generated_text", ""))
            
            # Estimate tokens used (simple estimation)
            tokens_used = len(generated_text.split())
            
            # Estimate confidence based on various factors
            confidence = self._estimate_confidence(generated_text)
            
            # Parse potential robot actions from the response
            parsed_actions = self._parse_robot_actions(generated_text)
            
            return LLMResponse(
                text=generated_text,
                model=request.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence=confidence,
                parsed_actions=parsed_actions
            )
        
        except requests.exceptions.RequestException as e:
            # Handle request errors
            error_time = time.time() - start_time
            return LLMResponse(
                text=f"Error: Unable to generate response - {str(e)}",
                model=request.model,
                tokens_used=0,
                processing_time=error_time,
                confidence=0.0
            )
        except Exception as e:
            # Handle other errors
            error_time = time.time() - start_time
            return LLMResponse(
                text=f"Error: Unexpected error occurred - {str(e)}",
                model=request.model,
                tokens_used=0,
                processing_time=error_time,
                confidence=0.0
            )
    
    async def generate_text_async(self, request: LLMRequest) -> LLMResponse:
        """Asynchronously generate text using the NIM service"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_text, request)
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate the confidence of the generated text"""
        # Simple confidence estimation based on text quality indicators
        # In a real implementation, this could use more sophisticated metrics
        
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Check for common indicators of low-quality responses
        low_quality_indicators = [
            "i don't know",
            "i am unable to",
            "i cannot",
            "sorry",
            "i'm sorry"
        ]
        
        text_lower = text.lower()
        quality_score = 1.0
        
        for indicator in low_quality_indicators:
            if indicator in text_lower:
                quality_score *= 0.5  # Reduce confidence if low-quality indicators are found
        
        # Length-based scoring (very short responses might be less confident)
        if len(text.split()) < 3:
            quality_score *= 0.7
        
        return max(0.1, min(1.0, quality_score))
    
    def _parse_robot_actions(self, text: str) -> List[Dict[str, Any]]:
        """Parse potential robot actions from the generated text"""
        actions = []
        
        # Look for command patterns in the text
        import re
        
        # Define patterns for common robot actions
        patterns = [
            (r"move to ([^,!.]+)", "navigation", {"location": r"move to ([^,!.]+)"}),
            (r"go to ([^,!.]+)", "navigation", {"location": r"go to ([^,!.]+)"}),
            (r"navigate to ([^,!.]+)", "navigation", {"location": r"navigate to ([^,!.]+)"}),
            (r"pick up ([^,!.]+)", "manipulation", {"object": r"pick up ([^,!.]+)"}),
            (r"grasp ([^,!.]+)", "manipulation", {"object": r"grasp ([^,!.]+)"}),
            (r"take ([^,!.]+)", "manipulation", {"object": r"take ([^,!.]+)"}),
            (r"turn (left|right)", "navigation", {"direction": r"turn (left|right)"}),
            (r"move (forward|backward)", "navigation", {"direction": r"move (forward|backward)"}),
            (r"stop", "navigation", {"command": "stop"}),
            (r"look at ([^,!.]+)", "perception", {"target": r"look at ([^,!.]+)"}),
            (r"find ([^,!.]+)", "perception", {"object": r"find ([^,!.]+)"})
        ]
        
        text_lower = text.lower()
        
        for pattern, action_type, param_pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    action = {
                        "type": action_type,
                        "raw_command": pattern,
                        "parameters": {}
                    }
                    
                    # Extract parameters based on the param_pattern
                    for param_name, param_regex in param_pattern.items():
                        param_match = re.search(param_regex, text_lower)
                        if param_match:
                            # If it's a group match, take the first group, otherwise take the whole match
                            if isinstance(param_match.groups(), tuple) and len(param_match.groups()) > 0:
                                action["parameters"][param_name] = param_match.groups()[0]
                            else:
                                action["parameters"][param_name] = param_match.group(0)
                    
                    actions.append(action)
        
        return actions


class ROS2LLMInterface:
    """
    Interface between LLM and ROS 2 systems for conversational robotics.
    This class handles communication between the LLM service and ROS 2 nodes.
    """
    
    def __init__(self, nim_client: NIMClient, robot_namespace: str = "/robot1"):
        self.nim_client = nim_client
        self.robot_namespace = robot_namespace
        self.is_running = False
        
        # ROS 2 publishers and subscribers
        self.response_publisher = None
        self.command_subscriber = None
        self.action_publisher = None
        
        # Conversation state
        self.conversation_history = []
        self.current_context = {}
        
        # Action execution status
        self.action_status = {}
    
    def initialize_ros(self):
        """Initialize ROS 2 communication"""
        rospy.init_node('llm_ros_interface', anonymous=True)
        
        # Publishers
        self.response_publisher = rospy.Publisher(
            f'{self.robot_namespace}/llm_response', 
            String, 
            queue_size=10
        )
        
        self.action_publisher = rospy.Publisher(
            f'{self.robot_namespace}/robot_action', 
            String, 
            queue_size=10
        )
        
        # Subscribers
        self.command_subscriber = rospy.Subscriber(
            f'{self.robot_namespace}/command_input', 
            String, 
            self.command_callback
        )
        
        rospy.loginfo("LLM-ROS interface initialized")
    
    def command_callback(self, data: String):
        """Handle incoming commands from other ROS nodes"""
        command = data.data
        rospy.loginfo(f"Received command: {command}")
        
        # Process the command with the LLM
        response = self.process_command(command)
        
        # Publish the response
        if self.response_publisher:
            self.response_publisher.publish(response.text)
        
        # Publish any parsed actions
        if response.parsed_actions:
            for action in response.parsed_actions:
                self.publish_action(action)
    
    def process_command(self, command: str, user_context: Dict[str, Any] = None) -> LLMResponse:
        """Process a natural language command using the LLM"""
        # Create a request with conversation context
        full_prompt = self._build_contextual_prompt(command, user_context)
        
        request = LLMRequest(
            prompt=full_prompt,
            model=self.nim_client.default_model,
            max_tokens=self.nim_client.default_max_tokens,
            temperature=self.nim_client.default_temperature,
            user_context=user_context or {}
        )
        
        # Get response from LLM
        response = self.nim_client.generate_text(request)
        
        # Add to conversation history
        self.conversation_history.append({
            "user": command,
            "assistant": response.text,
            "timestamp": time.time()
        })
        
        # Keep only the last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def _build_contextual_prompt(self, command: str, user_context: Dict[str, Any] = None) -> str:
        """Build a prompt that includes conversation context"""
        # Start with system context
        system_context = (
            "You are a helpful robotic assistant. "
            "Your responses should be clear, concise, and actionable. "
            "If the user gives you a command, respond with a plan of actions "
            "that the robot should take, using simple language."
        )
        
        # Add recent conversation history
        history_context = ""
        if self.conversation_history:
            history_context = "Recent conversation:\n"
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                history_context += f"User: {exchange['user']}\n"
                history_context += f"Assistant: {exchange['assistant']}\n"
        
        # Add user-specific context if provided
        user_context_str = ""
        if user_context:
            user_context_str = f"User context: {user_context}\n"
        
        # Construct the full prompt
        full_prompt = (
            f"{system_context}\n\n"
            f"{user_context_str}\n"
            f"{history_context}\n"
            f"User command: {command}\n"
            f"Assistant response (with actionable steps):"
        )
        
        return full_prompt
    
    def publish_action(self, action: Dict[str, Any]):
        """Publish a robot action to the ROS 2 system"""
        if self.action_publisher:
            action_json = json.dumps(action)
            self.action_publisher.publish(action_json)
            rospy.loginfo(f"Published action: {action_json}")
    
    def start_listening(self):
        """Start the ROS 2 node and begin listening for commands"""
        if not rospy.core.is_initialized():
            self.initialize_ros()
        
        self.is_running = True
        rospy.loginfo("LLM-ROS interface started listening")
        
        # Keep the node running
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("LLM-ROS interface stopped")
            self.is_running = False
    
    def stop_listening(self):
        """Stop the ROS 2 node"""
        self.is_running = False
        rospy.signal_shutdown("LLM-ROS interface stopped")


class LLMActionExecutor:
    """
    Executes actions parsed from LLM responses in the ROS 2 environment
    """
    
    def __init__(self, robot_namespace: str = "/robot1"):
        self.robot_namespace = robot_namespace
        self.action_subscriber = None
        self.status_publisher = None
        
        # Action execution mapping
        self.action_handlers = {
            "navigation": self._handle_navigation,
            "manipulation": self._handle_manipulation,
            "perception": self._handle_perception
        }
    
    def initialize_ros(self):
        """Initialize ROS 2 communication for action execution"""
        if not rospy.core.is_initialized():
            rospy.init_node('llm_action_executor', anonymous=True)
        
        # Subscriber for actions from LLM
        self.action_subscriber = rospy.Subscriber(
            f'{self.robot_namespace}/robot_action',
            String,
            self.action_callback
        )
        
        # Publisher for action status
        self.status_publisher = rospy.Publisher(
            f'{self.robot_namespace}/action_status',
            String,
            queue_size=10
        )
        
        rospy.loginfo("LLM Action Executor initialized")
    
    def action_callback(self, data: String):
        """Handle incoming actions from the LLM"""
        try:
            action_data = json.loads(data.data)
            action_type = action_data.get("type")
            
            rospy.loginfo(f"Received action: {action_type}")
            
            if action_type in self.action_handlers:
                # Execute the action
                success = self.action_handlers[action_type](action_data)
                
                # Publish status
                status = {
                    "action": action_type,
                    "success": success,
                    "timestamp": time.time()
                }
                
                if self.status_publisher:
                    self.status_publisher.publish(json.dumps(status))
            else:
                rospy.logerr(f"Unknown action type: {action_type}")
        
        except json.JSONDecodeError:
            rospy.logerr("Invalid action JSON received")
        except Exception as e:
            rospy.logerr(f"Error processing action: {e}")
    
    def _handle_navigation(self, action_data: Dict[str, Any]) -> bool:
        """Handle navigation actions"""
        try:
            location = action_data.get("parameters", {}).get("location")
            direction = action_data.get("parameters", {}).get("direction")
            
            if location:
                rospy.loginfo(f"Moving to location: {location}")
                # In a real system, this would call navigation services
                # Example: call_move_base(location)
                return True
            elif direction:
                rospy.loginfo(f"Moving {direction}")
                # In a real system, this would call movement services
                # Example: call_move_direction(direction)
                return True
            else:
                rospy.logerr("Navigation action missing location or direction")
                return False
        except Exception as e:
            rospy.logerr(f"Error in navigation handler: {e}")
            return False
    
    def _handle_manipulation(self, action_data: Dict[str, Any]) -> bool:
        """Handle manipulation actions"""
        try:
            obj = action_data.get("parameters", {}).get("object")
            
            if obj:
                rospy.loginfo(f"Attempting to manipulate object: {obj}")
                # In a real system, this would call manipulation services
                # Example: call_grasp_object(obj)
                return True
            else:
                rospy.logerr("Manipulation action missing object")
                return False
        except Exception as e:
            rospy.logerr(f"Error in manipulation handler: {e}")
            return False
    
    def _handle_perception(self, action_data: Dict[str, Any]) -> bool:
        """Handle perception actions"""
        try:
            target = action_data.get("parameters", {}).get("target") or \
                    action_data.get("parameters", {}).get("object")
            
            if target:
                rospy.loginfo(f"Looking for: {target}")
                # In a real system, this would call perception services
                # Example: call_find_object(target)
                return True
            else:
                rospy.logerr("Perception action missing target")
                return False
        except Exception as e:
            rospy.logerr(f"Error in perception handler: {e}")
            return False
    
    def start_execution(self):
        """Start listening for actions to execute"""
        if not rospy.core.is_initialized():
            self.initialize_ros()
        
        rospy.loginfo("LLM Action Executor started")
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("LLM Action Executor stopped")


# Example usage and testing functions
def test_nim_integration():
    """Test function to verify NIM integration works"""
    # Create a mock NIM client (in a real system, this would connect to actual NIM service)
    nim_client = NIMClient(nim_endpoint="http://localhost:8000")
    
    # Create a test request
    request = LLMRequest(
        prompt="What is 2+2? Respond with just the number.",
        model=LLMModelType.LLAMA2_7B,
        max_tokens=10,
        temperature=0.1  # Low temperature for consistent results
    )
    
    # Get response
    response = nim_client.generate_text(request)
    
    print(f"Prompt: {request.prompt}")
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Processing time: {response.processing_time:.2f}s")
    print(f"Confidence: {response.confidence:.2f}")
    
    return response


def main():
    """Main function to demonstrate LLM-ROS integration"""
    # Initialize NIM client
    nim_client = NIMClient(
        nim_endpoint="http://localhost:8000",  # Replace with actual NIM endpoint
        api_key=None  # Add API key if required
    )
    
    # Initialize the ROS interface
    ros_interface = ROS2LLMInterface(nim_client)
    
    # Initialize action executor
    action_executor = LLMActionExecutor()
    
    # In a real system, you would run these in separate threads/processes
    # For demonstration, we'll show how they would be started:
    print("LLM-ROS integration initialized")
    print("To run: ros_interface.start_listening() and action_executor.start_execution()")


if __name__ == "__main__":
    main()