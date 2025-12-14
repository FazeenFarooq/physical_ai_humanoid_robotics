"""
Vision-Language-Action (VLA) Model Integration for the Physical AI & Humanoid Robotics Course.

This module provides integration of vision and language models for embodied AI systems,
following the course's emphasis on Vision-Language-Action cognitive stack integration.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import os
import logging
from PIL import Image


@dataclass
class VLAModelConfig:
    """Configuration for Vision-Language-Action model."""
    vision_model_name: str = "google/vit-base-patch16-224"
    language_model_name: str = "bert-base-uncased"
    projection_dim: int = 512
    max_seq_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_vision_backbone: bool = False
    freeze_language_backbone: bool = False
    fusion_method: str = "cross_attention"  # Options: "cross_attention", "concat", "late_fusion"


class VisionEncoder(nn.Module):
    """Vision encoder for processing visual input."""
    
    def __init__(self, model_name: str, projection_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        
        # Load a Vision Transformer model
        from transformers import ViTModel
        self.vision_model = ViTModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Projection layer to map vision features to common space
        self.projection = nn.Linear(self.vision_model.config.hidden_size, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder."""
        # Get vision embeddings
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_embeddings = vision_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to common dimension
        projected_embeddings = self.projection(vision_embeddings)  # [batch_size, seq_len, projection_dim]
        projected_embeddings = self.layer_norm(projected_embeddings)
        projected_embeddings = self.dropout(projected_embeddings)
        
        return projected_embeddings  # [batch_size, seq_len, projection_dim]


class LanguageEncoder(nn.Module):
    """Language encoder for processing text input."""
    
    def __init__(self, model_name: str, projection_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained language model (BERT-based)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layer to map language features to common space
        self.projection = nn.Linear(self.language_model.config.hidden_size, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through language encoder."""
        # Get language embeddings
        language_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        language_embeddings = language_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to common dimension
        projected_embeddings = self.projection(language_embeddings)  # [batch_size, seq_len, projection_dim]
        projected_embeddings = self.layer_norm(projected_embeddings)
        projected_embeddings = self.dropout(projected_embeddings)
        
        return projected_embeddings  # [batch_size, seq_len, projection_dim]


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing vision and language."""
    
    def __init__(self, projection_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 4, projection_dim),
            nn.Dropout(0.1)
        )
        
        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.layer_norm2 = nn.LayerNorm(projection_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention between vision and language features."""
        # Attention: language attending to vision
        attended_lang, _ = self.multihead_attn(
            query=language_features,
            key=vision_features,
            value=vision_features
        )
        
        # Residual connection and layer norm
        attended_lang = self.layer_norm1(attended_lang + language_features)
        
        # Feed-forward
        ff_lang = self.feed_forward(attended_lang)
        lang_output = self.layer_norm2(ff_lang + attended_lang)
        
        # Attention: vision attending to language
        attended_vis, _ = self.multihead_attn(
            query=vision_features,
            key=lang_output,
            value=lang_output
        )
        
        # Residual connection and layer norm
        attended_vis = self.layer_norm1(attended_vis + vision_features)
        
        # Feed-forward
        ff_vis = self.feed_forward(attended_vis)
        vis_output = self.layer_norm2(ff_vis + attended_vis)
        
        return vis_output, lang_output


class ActionDecoder(nn.Module):
    """Decoder that generates action commands from fused representations."""
    
    def __init__(self, input_dim: int, num_actions: int = 64, hidden_dim: int = 512):
        super().__init__()
        
        # Action prediction head
        self.action_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Continuous action parameters (e.g., arm positions, gripper values)
        self.param_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 10)  # 10 continuous parameters
        )
        
        # Action probability predictor
        self.prob_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode fused features to action commands."""
        # Predict discrete action
        action_logits = self.action_predictor(fused_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Predict continuous action parameters
        action_params = torch.tanh(self.param_predictor(fused_features))  # Clamp between -1 and 1
        
        # Predict action probability/confidence
        action_prob = self.prob_predictor(fused_features)
        
        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "action_params": action_params,
            "action_confidence": action_prob
        }


class VLAModel(nn.Module):
    """Vision-Language-Action Model that integrates all components."""
    
    def __init__(self, config: VLAModelConfig):
        super().__init__()
        
        self.config = config
        self.device = config.device
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=config.vision_model_name,
            projection_dim=config.projection_dim,
            freeze_backbone=config.freeze_vision_backbone
        )
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            model_name=config.language_model_name,
            projection_dim=config.projection_dim,
            freeze_backbone=config.freeze_language_backbone
        )
        
        # Cross-modal fusion
        if config.fusion_method == "cross_attention":
            self.fusion_layer = CrossModalAttention(
                projection_dim=config.projection_dim
            )
        else:
            # For other fusion methods, we could implement them here
            self.fusion_layer = CrossModalAttention(
                projection_dim=config.projection_dim
            )
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            input_dim=config.projection_dim * 2,  # Concatenated vision and language features
            num_actions=64,  # Define number of possible actions
            hidden_dim=config.projection_dim * 2
        )
        
        # Global average pooling for sequence-level representations
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for the model."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode visual input."""
        return self.vision_encoder(pixel_values)
    
    def encode_language(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode language input."""
        return self.language_encoder(input_ids, attention_mask)
    
    def fuse_modalities(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse vision and language features."""
        fused_vis, fused_lang = self.fusion_layer(vision_features, language_features)
        
        # Concatenate the fused representations
        # Pool the sequences to get fixed-size representations
        pooled_vis = self.global_pool(fused_vis.transpose(-2, -1)).squeeze(-1)  # [B, D]
        pooled_lang = self.global_pool(fused_lang.transpose(-2, -1)).squeeze(-1)  # [B, D]
        
        # Concatenate vision and language features
        concatenated = torch.cat([pooled_vis, pooled_lang], dim=-1)  # [B, 2*D]
        
        return concatenated
    
    def decode_actions(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode fused features to actions."""
        return self.action_decoder(fused_features)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the entire VLA model."""
        # Encode vision
        vision_features = self.encode_vision(pixel_values)
        
        # Encode language
        language_features = self.encode_language(input_ids, attention_mask)
        
        # Fuse modalities
        fused_features = self.fuse_modalities(vision_features, language_features)
        
        # Decode actions
        action_outputs = self.decode_actions(fused_features)
        
        return {
            "vision_features": vision_features,
            "language_features": language_features,
            "fused_features": fused_features,
            **action_outputs
        }
    
    def predict_action(
        self, 
        image: Union[np.ndarray, torch.Tensor], 
        text: str
    ) -> Dict[str, Any]:
        """Predict action from image and text input."""
        self.eval()
        
        # Process image
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Convert float array to uint8
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Apply image transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(pil_image).unsqueeze(0)  # Add batch dimension
        else:
            pixel_values = image.unsqueeze(0) if len(image.shape) == 3 else image
        
        # Process text
        tokenization_result = self.language_encoder.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        input_ids = tokenization_result["input_ids"]
        attention_mask = tokenization_result["attention_mask"]
        
        # Move to device
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_ids, attention_mask)
        
        # Extract action prediction
        action_probs = outputs["action_probs"][0].cpu().numpy()
        action_params = outputs["action_params"][0].cpu().numpy()
        action_confidence = outputs["action_confidence"][0].cpu().numpy()
        
        # Get predicted action
        predicted_action_idx = np.argmax(action_probs)
        
        return {
            "predicted_action_idx": predicted_action_idx,
            "action_probabilities": action_probs,
            "action_parameters": action_params,
            "action_confidence": float(action_confidence),
            "raw_outputs": outputs
        }
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text."""
        tokenization_result = self.language_encoder.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        input_ids = tokenization_result["input_ids"].to(self.device)
        attention_mask = tokenization_result["attention_mask"].to(self.device)
        
        with torch.no_grad():
            language_features = self.encode_language(input_ids, attention_mask)
        
        return language_features
    
    def get_image_embedding(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get embedding for image."""
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Convert float array to uint8
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Apply image transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(pil_image).unsqueeze(0)  # Add batch dimension
        else:
            pixel_values = image.unsqueeze(0) if len(image.shape) == 3 else image
        
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            vision_features = self.encode_vision(pixel_values)
        
        return vision_features


class VLAProcessor:
    """Processor for handling VLA model inputs and outputs."""
    
    def __init__(self, model: VLAModel):
        self.model = model
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger("VLAProcessor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def process_perception_input(
        self, 
        rgb_image: np.ndarray, 
        instruction: str
    ) -> Dict[str, Any]:
        """Process perception input and generate action."""
        self.logger.info(f"Processing perception input: {instruction}")
        
        # Get action prediction
        result = self.model.predict_action(rgb_image, instruction)
        
        # Convert action index to meaningful action
        action_name = self._action_index_to_name(result["predicted_action_idx"])
        
        processed_result = {
            "action_name": action_name,
            "action_index": result["predicted_action_idx"],
            "action_probability": float(result["action_probabilities"][result["predicted_action_idx"]]),
            "action_parameters": result["action_parameters"].tolist(),
            "action_confidence": result["action_confidence"],
            "instruction": instruction,
            "status": "success"
        }
        
        self.logger.info(f"Generated action: {action_name} with confidence: {result['action_confidence']:.2f}")
        
        return processed_result
    
    def _action_index_to_name(self, action_idx: int) -> str:
        """Map action index to meaningful name."""
        # This is a simplified mapping
        # In a real implementation, this would correspond to actual robot actions
        action_names = {
            0: "move_forward",
            1: "move_backward", 
            2: "turn_left",
            3: "turn_right",
            4: "grasp_object",
            5: "release_object",
            6: "raise_arm",
            7: "lower_arm",
            8: "wave",
            9: "point",
            # Add more actions as needed
        }
        
        return action_names.get(action_idx, f"action_{action_idx}")
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        # Flatten embeddings if necessary
        if len(embedding1.shape) > 2:
            embedding1 = embedding1.view(embedding1.shape[0], -1)
        if len(embedding2.shape) > 2:
            embedding2 = embedding2.view(embedding2.shape[0], -1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=-1)
        return float(cos_sim.mean().cpu().numpy())


def create_default_vla_model() -> VLAModel:
    """Create a VLA model with default configuration."""
    config = VLAModelConfig()
    return VLAModel(config)


def example_usage():
    """Example of how to use the VLA model integration."""
    print("Vision-Language-Action Model Integration Example")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU (may be slow)")
    
    # Create a VLA model with default configuration
    try:
        model = create_default_vla_model()
        print(f"VLA Model created successfully on {model.device}")
        
        # Create processor
        processor = VLAProcessor(model)
        
        # Simulate an RGB image (224x224x3)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Simulate an instruction
        instruction = "Pick up the red cup on the table"
        
        print(f"\nProcessing instruction: '{instruction}'")
        print("Image shape:", image.shape)
        
        # Process the input
        result = processor.process_perception_input(image, instruction)
        
        print(f"\nAction prediction: {result['action_name']}")
        print(f"Action confidence: {result['action_confidence']:.3f}")
        print(f"Action probability: {result['action_probability']:.3f}")
        print(f"Action parameters: {result['action_parameters'][:5]}...")  # Show first 5 params
        
        # Example of getting embeddings
        print("\nGetting embeddings...")
        img_embedding = model.get_image_embedding(image)
        txt_embedding = model.get_text_embedding(instruction)
        
        similarity = processor.compute_similarity(img_embedding, txt_embedding)
        print(f"Image-text similarity: {similarity:.3f}")
        
    except ImportError as e:
        print(f"Missing dependencies for VLA model: {e}")
        print("Please install required packages: transformers, torchvision")
    except Exception as e:
        print(f"Error creating or using VLA model: {e}")
        print("This may be due to network connectivity issues or missing model files")
        
        # Create a dummy processor for demonstration
        print("\nDemonstration using dummy model...")
        print("If properly implemented, this would:")
        print("- Take an image and a text instruction")
        print("- Process both modalities through Vision and Language encoders")
        print("- Fuse the modalities using cross-attention")
        print("- Generate an action prediction based on the fused representation")
        print("- Output action name, confidence, and parameters")


if __name__ == "__main__":
    example_usage()