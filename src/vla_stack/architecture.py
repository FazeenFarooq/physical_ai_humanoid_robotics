"""
Vision-Language-Action (VLA) Cognitive Stack Architecture for the Physical AI & Humanoid Robotics Course.

This module implements the complete cognitive stack that integrates vision, language, 
and action capabilities for embodied AI systems, following the course's emphasis on 
unified cognitive architectures for robotic systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import threading
import time
import logging
from enum import Enum


class CognitiveLayer(Enum):
    """Different layers in the cognitive stack."""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTION = "action"
    LEARNING = "learning"


@dataclass
class CognitiveState:
    """Represents the state of the cognitive system."""
    layer: CognitiveLayer
    timestamp: float
    activation: float  # Activation level (0-1)
    features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class PerceptionLayer(nn.Module):
    """Bottom layer handling raw sensory input processing."""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size after convolutions
        conv_output_size = self._get_conv_output_size((input_channels, 224, 224))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
        
        self.activation = 0.0
    
    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate the output size after convolution layers."""
        x = torch.randn(1, *input_shape)
        x = self.conv_layers[:6](x)  # Apply conv layers to get flattened size
        return int(np.prod(x.size()[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process visual input."""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        # Update activation level based on input strength
        self.activation = torch.mean(torch.abs(x)).item()
        
        return x


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant information."""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.activation = 0.0
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism."""
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value
        )
        
        # Residual connection and normalization
        output = self.layer_norm(attn_output + query)
        output = self.dropout(output)
        
        # Update activation level
        self.activation = torch.mean(torch.abs(output)).item()
        
        return output, attn_weights


class MemoryLayer(nn.Module):
    """Episodic and working memory system."""
    
    def __init__(self, memory_size: int = 100, feature_dim: int = 512):
        super().__init__()
        
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Episodic memory storage (simplified as a register)
        self.episodic_memory = nn.Parameter(
            torch.randn(memory_size, feature_dim) * 0.1,
            requires_grad=False
        )
        
        # Working memory (limited capacity)
        self.working_memory_capacity = 10
        self.working_memory_items: List[Tuple[torch.Tensor, float]] = []  # (features, timestamp)
        
        self.activation = 0.0
    
    def store_episode(self, features: torch.Tensor, timestamp: Optional[float] = None):
        """Store an episode in episodic memory."""
        if timestamp is None:
            timestamp = time.time()
        
        # Shift memory buffer and add new item
        self.episodic_memory = torch.roll(self.episodic_memory, shifts=1, dims=0)
        self.episodic_memory[0] = features.detach()
    
    def store_working_item(self, features: torch.Tensor, priority: float = 1.0):
        """Store an item in working memory."""
        if len(self.working_memory_items) >= self.working_memory_capacity:
            # Replace lowest priority item or oldest
            self.working_memory_items.pop(0)
        
        self.working_memory_items.append((features.detach(), priority))
        self.working_memory_items.sort(key=lambda x: x[1], reverse=True)  # Sort by priority
    
    def retrieve_similar(self, query: torch.Tensor, top_k: int = 3) -> torch.Tensor:
        """Retrieve similar memories to the query."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Calculate similarity with episodic memory
        similarities = torch.matmul(query, self.episodic_memory.t())
        top_indices = torch.topk(similarities, k=min(top_k, self.episodic_memory.size(0)), dim=1).indices
        
        # Return retrieved memories
        retrieved = self.episodic_memory[top_indices.squeeze()].mean(dim=0)
        return retrieved
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant information from memory."""
        retrieved = self.retrieve_similar(query)
        
        # Update activation based on memory match
        if retrieved.numel() > 0:
            self.activation = torch.mean(torch.abs(retrieved)).item()
        
        return retrieved


class ReasoningLayer(nn.Module):
    """Logical reasoning and inference engine."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Neural reasoning module
        self.reasoning_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # Combine current state and memory
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        
        self.activation = 0.0
    
    def forward(self, current_features: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        """Perform reasoning over current and memory features."""
        # Combine current state with memory
        combined = torch.cat([current_features, memory_features], dim=-1)
        
        # Apply reasoning
        reasoning_output = self.reasoning_net(combined)
        
        # Update activation
        self.activation = torch.mean(reasoning_output).item()
        
        return reasoning_output


class PlanningLayer(nn.Module):
    """Hierarchical planning and goal decomposition."""
    
    def __init__(self, feature_dim: int, max_plan_length: int = 10):
        super().__init__()
        
        self.max_plan_length = max_plan_length
        
        # Plan representation network
        self.plan_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Action prediction head
        self.action_predictor = nn.Linear(feature_dim, feature_dim)
        
        self.activation = 0.0
    
    def forward(self, reasoning_features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate a plan from reasoning output."""
        # Expand features to sequence format for LSTM
        expanded_features = reasoning_features.unsqueeze(1).expand(-1, self.max_plan_length, -1)
        
        # Generate plan
        plan_sequence, _ = self.plan_encoder(expanded_features)
        
        # Predict actions for each timestep
        action_predictions = self.action_predictor(plan_sequence)
        
        # Convert to list of actions
        action_list = [action_predictions[:, i, :] for i in range(self.max_plan_length)]
        
        # Update activation
        self.activation = torch.mean(torch.abs(action_predictions)).item()
        
        return plan_sequence, action_list


class ActionSelectionLayer(nn.Module):
    """Selects and executes actions based on plan."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Action selection network
        self.action_selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 64),  # 64 possible actions
            nn.Softmax(dim=-1)
        )
        
        self.activation = 0.0
    
    def forward(self, plan_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Select action from plan features."""
        action_probs = self.action_selector(plan_features)
        
        # Select action with highest probability
        selected_action_idx = torch.argmax(action_probs, dim=-1)
        
        # Update activation
        self.activation = torch.max(action_probs).item()
        
        return action_probs, selected_action_idx.item()


class VLACognitiveStack(nn.Module):
    """Complete VLA cognitive stack integrating all cognitive layers."""
    
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 512,
        num_heads: int = 8,
        memory_size: int = 100,
        max_plan_length: int = 10
    ):
        super().__init__()
        
        # Initialize cognitive layers
        self.perception = PerceptionLayer(input_channels, feature_dim)
        self.attention = AttentionLayer(feature_dim, num_heads)
        self.memory = MemoryLayer(memory_size, feature_dim)
        self.reasoning = ReasoningLayer(feature_dim, feature_dim * 2)
        self.planning = PlanningLayer(feature_dim, max_plan_length)
        self.action_selection = ActionSelectionLayer(feature_dim)
        
        self.feature_dim = feature_dim
        self.state_history: List[CognitiveState] = []
        
        # Learning components
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the cognitive stack."""
        logger = logging.getLogger("VLACognitiveStack")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def forward(
        self, 
        visual_input: torch.Tensor, 
        language_input: Optional[torch.Tensor] = None,
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the complete cognitive pipeline."""
        start_time = time.time()
        
        # 1. PERCEPTION: Process visual input
        perceptual_features = self.perception(visual_input)
        self._record_state(CognitiveLayer.PERCEPTION, perceptual_features)
        
        # 2. ATTENTION: Focus on relevant features
        # For now, we'll use the same features as key and value
        attended_features, attention_weights = self.attention(
            query=perceptual_features.unsqueeze(1),
            key=perceptual_features.unsqueeze(1),
            value=perceptual_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        self._record_state(CognitiveLayer.ATTENTION, attended_features, attention_weights)
        
        # 3. MEMORY: Retrieve relevant information
        memory_features = self.memory(attended_features)
        self._record_state(CognitiveLayer.MEMORY, memory_features)
        
        # 4. REASONING: Apply logical reasoning
        reasoning_output = self.reasoning(attended_features, memory_features)
        self._record_state(CognitiveLayer.REASONING, reasoning_output)
        
        # 5. PLANNING: Generate action plan
        plan_sequence, action_list = self.planning(reasoning_output)
        self._record_state(CognitiveLayer.PLANNING, plan_sequence.mean(dim=1))
        
        # 6. ACTION SELECTION: Choose specific actions
        action_probs, selected_action = self.action_selection(plan_sequence[:, 0, :])  # Use first timestep
        self._record_state(CognitiveLayer.ACTION, action_probs)
        
        total_time = time.time() - start_time
        
        return {
            "selected_action": selected_action,
            "action_probabilities": action_probs,
            "plan": action_list,
            "reasoning_output": reasoning_output,
            "processing_time": total_time,
            "activation_levels": {
                "perception": self.perception.activation,
                "attention": self.attention.activation,
                "memory": self.memory.activation,
                "reasoning": self.reasoning.activation,
                "planning": self.planning.activation,
                "action_selection": self.action_selection.activation
            }
        }
    
    def _record_state(
        self, 
        layer: CognitiveLayer, 
        features: torch.Tensor, 
        attention_weights: Optional[torch.Tensor] = None
    ):
        """Record the state of a cognitive layer."""
        state = CognitiveState(
            layer=layer,
            timestamp=time.time(),
            activation=torch.mean(torch.abs(features)).item(),
            features=features.detach() if features.requires_grad else features,
            attention_weights=attention_weights,
            metadata={"layer_name": layer.value}
        )
        self.state_history.append(state)
        
        # Keep only last 100 states to avoid memory buildup
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def update_memory(self, features: torch.Tensor, timestamp: Optional[float] = None):
        """Update memory with new information."""
        self.memory.store_episode(features, timestamp)
        self.logger.info(f"Updated memory with new episode at {timestamp or time.time()}")
    
    def get_cognitive_state(self) -> List[CognitiveState]:
        """Get the recent cognitive state history."""
        return self.state_history.copy()
    
    def learn_from_interaction(
        self, 
        state_before: Dict[str, Any], 
        action_taken: int, 
        reward: float, 
        state_after: Dict[str, Any]
    ):
        """Update the cognitive stack based on interaction outcomes."""
        # This would implement learning algorithms like reinforcement learning
        # For now, we'll log the learning update
        self.logger.info(f"Learning from action {action_taken} with reward {reward}")
        
        # In a real implementation, this would adjust internal parameters
        # based on the reward signal
        pass


class CognitiveController:
    """Controller for managing the cognitive stack."""
    
    def __init__(self, cognitive_stack: VLACognitiveStack):
        self.stack = cognitive_stack
        self.active = False
        self.thread = None
        self.input_queue = []
        self.output_queue = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the controller."""
        logger = logging.getLogger("CognitiveController")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start(self):
        """Start the cognitive controller."""
        self.active = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.logger.info("Cognitive controller started")
    
    def stop(self):
        """Stop the cognitive controller."""
        self.active = False
        if self.thread:
            self.thread.join()
        self.logger.info("Cognitive controller stopped")
    
    def _run(self):
        """Main cognitive processing loop."""
        while self.active:
            if self.input_queue:
                # Process next input
                visual_input, language_input, task_desc = self.input_queue.pop(0)
                
                try:
                    result = self.stack(
                        visual_input=visual_input,
                        language_input=language_input,
                        task_description=task_desc
                    )
                    
                    self.output_queue.append(result)
                    
                    # Update memory with the result
                    if "reasoning_output" in result:
                        self.stack.update_memory(result["reasoning_output"])
                
                except Exception as e:
                    self.logger.error(f"Error in cognitive processing: {e}")
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
    
    def submit_input(
        self, 
        visual_input: torch.Tensor, 
        language_input: Optional[torch.Tensor] = None,
        task_description: Optional[str] = None
    ):
        """Submit input for cognitive processing."""
        self.input_queue.append((visual_input, language_input, task_description))
    
    def get_output(self) -> Optional[Dict[str, Any]]:
        """Get the next output from the cognitive stack."""
        if self.output_queue:
            return self.output_queue.pop(0)
        return None


def create_default_vla_stack() -> VLACognitiveStack:
    """Create a VLA cognitive stack with default configuration."""
    return VLACognitiveStack(
        input_channels=3,
        feature_dim=512,
        num_heads=8,
        memory_size=100,
        max_plan_length=10
    )


def example_usage():
    """Example of how to use the VLA cognitive stack."""
    print("VLA Cognitive Stack Architecture Example")
    print("=" * 50)
    
    # Create the cognitive stack
    cognitive_stack = create_default_vla_stack()
    print(f"Cognitive stack created with layers: {[layer.value for layer in CognitiveLayer]}")
    
    # Create a sample visual input (batch_size=1, channels=3, height=224, width=224)
    sample_visual_input = torch.randn(1, 3, 224, 224)
    
    # Process the input through the cognitive stack
    print("\nProcessing sample input through cognitive stack...")
    result = cognitive_stack(visual_input=sample_visual_input)
    
    print(f"Selected action: {result['selected_action']}")
    print(f"Processing time: {result['processing_time']:.4f}s")
    print(f"Activation levels: {result['activation_levels']}")
    
    # Show cognitive state history
    state_history = cognitive_stack.get_cognitive_state()
    print(f"\nCognitive state history has {len(state_history)} entries")
    if state_history:
        latest_state = state_history[-1]
        print(f"Latest cognitive state: {latest_state.layer.value}")
        print(f"Activation: {latest_state.activation:.3f}")
    
    # Example with controller
    print("\nTesting cognitive controller...")
    controller = CognitiveController(cognitive_stack)
    controller.start()
    
    # Submit a sample input
    controller.submit_input(sample_visual_input)
    
    # Wait for output and get it
    time.sleep(0.1)  # Allow processing time
    output = controller.get_output()
    
    if output:
        print("Controller output received:")
        print(f"  Selected action: {output['selected_action']}")
        print(f"  Processing time: {output['processing_time']:.4f}s")
    else:
        print("No output received from controller")
    
    # Stop controller
    controller.stop()
    
    print("\nVLA cognitive stack example completed successfully!")


if __name__ == "__main__":
    example_usage()