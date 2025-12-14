"""
Natural Language Understanding (NLU) for the Physical AI & Humanoid Robotics Course.

This module provides tools for understanding and interpreting natural language input
in the context of robotics applications, enabling robots to comprehend human instructions
and engage in meaningful conversations.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np
import logging


@dataclass
class Intent:
    """Represents an identified intent from natural language input."""
    name: str
    confidence: float
    parameters: Dict[str, Any]
    entities: List[Dict[str, Any]]


@dataclass
class ParsedUtterance:
    """Represents a parsed natural language utterance."""
    original_text: str
    intents: List[Intent]
    entities: List[Dict[str, Any]]
    tokens: List[str]
    lemmatized_tokens: List[str]
    pos_tags: List[Tuple[str, str]]
    dependencies: List[Tuple[str, str, str]]  # (token, relation, head)
    timestamp: float


class NLUPipeline:
    """Main pipeline for natural language understanding."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.intent_classifier = None
        self.entity_extractor = None
        self.nlp_model = None
        self.domain_actions = {}
        
        try:
            # Load spaCy model for linguistic processing
            self.nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy 'en_core_web_sm' model not found. "
                              "Install with: python -m spacy download en_core_web_sm")
        
        try:
            # Initialize Hugging Face pipeline for named entity recognition
            self.entity_extractor = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize NER pipeline: {e}")
        
        # Define domain-specific actions and their mappings
        self._initialize_domain_actions()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the NLU pipeline."""
        logger = logging.getLogger("NLUPipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_domain_actions(self):
        """Initialize domain-specific actions for robotics."""
        self.domain_actions = {
            # Movement actions
            "move_to_location": {
                "synonyms": ["go to", "move to", "navigate to", "go", "move", "walk", "travel"],
                "templates": [
                    r"(move|go|navigate|walk|travel)\s+(to|toward|towards)\s+(.+)",
                    r"(go|move)\s+(.+) (please|now)?",
                    r"please (.+) (to|toward|towards) (.+)"
                ]
            },
            # Grasping actions
            "grasp_object": {
                "synonyms": ["pick up", "grasp", "take", "grab", "hold", "get"],
                "templates": [
                    r"(pick up|grasp|take|grab|hold|get)\s+(.+) (from the (.+))?",
                    r"get me the (.+)",
                    r"can you pick up the (.+)"
                ]
            },
            # Manipulation actions
            "manipulate_object": {
                "synonyms": ["move", "push", "pull", "lift", "place", "set down"],
                "templates": [
                    r"(move|push|pull|lift|place|set down)\s+(.+) (to|on|onto)\s+(.+)",
                    r"move the (.+) (to|on|onto) (.+)"
                ]
            },
            # Navigation actions
            "navigate_to_object": {
                "synonyms": ["go to", "approach", "navigate to", "move toward"],
                "templates": [
                    r"go to the (.+)",
                    r"navigate to the (.+)",
                    r"approach the (.+)"
                ]
            },
            # Social actions
            "greet_person": {
                "synonyms": ["say hello", "greet", "hello", "hi", "good morning", "good evening"],
                "templates": [
                    r"(say hello|greet|hello|hi|good morning|good evening) (to .+)?",
                    r"can you say hi (to .+)?"
                ]
            },
            "follow_person": {
                "synonyms": ["follow", "come with", "stay with", "accompany"],
                "templates": [
                    r"follow (.+)",
                    r"come with (.+)",
                    r"stay with (.+)"
                ]
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Expand contractions (simple expansion)
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        if self.entity_extractor:
            try:
                ner_results = self.entity_extractor(text)
                for entity in ner_results:
                    entities.append({
                        "text": entity["word"],
                        "label": entity["entity_group"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "confidence": entity["score"]
                    })
            except Exception as e:
                self.logger.error(f"Error extracting entities: {e}")
        
        # Also use spaCy if available
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 1.0  # spaCy doesn't provide confidence scores in this way
                    })
            except Exception as e:
                self.logger.error(f"Error extracting entities with spaCy: {e}")
        
        return entities
    
    def identify_intent(self, text: str) -> List[Intent]:
        """Identify intent from text using regex patterns and domain knowledge."""
        intents = []
        
        # Process text
        processed_text = self.preprocess_text(text)
        
        # Check for each action in domain
        for action_name, action_info in self.domain_actions.items():
            confidence = 0
            parameters = {}
            matched_entities = []
            
            # Check synonyms
            for synonym in action_info["synonyms"]:
                if synonym.lower() in processed_text:
                    confidence = 0.7  # Base confidence if synonym matches
                    break
            
            # Check templates
            for template in action_info["templates"]:
                match = re.search(template, processed_text)
                if match:
                    # Higher confidence if template matches
                    confidence = 0.85
                    groups = match.groups()
                    
                    # Extract objects and locations from the match
                    for i, group in enumerate(groups):
                        if group and i == 0:  # First group usually the object
                            parameters["object"] = group.strip()
                        elif group and i == 1:  # Second group often the location
                            parameters["location"] = group.strip()
                        elif group and i == 2:  # Third group for additional info
                            parameters["destination"] = group.strip()
            
            if confidence > 0:
                # Extract entities for this intent
                extracted_entities = self.extract_entities(text)
                
                intents.append(Intent(
                    name=action_name,
                    confidence=confidence,
                    parameters=parameters,
                    entities=extracted_entities
                ))
        
        # Sort intents by confidence
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no specific intent matched, classify as general
        if not intents:
            intents.append(Intent(
                name="general_query",
                confidence=0.5,
                parameters={"text": processed_text},
                entities=self.extract_entities(text)
            ))
        
        return intents
    
    def parse_utterance(self, text: str) -> ParsedUtterance:
        """Parse a natural language utterance into structured components."""
        import time
        
        # Tokenization and linguistic analysis (if spaCy is available)
        tokens = []
        lemmatized_tokens = []
        pos_tags = []
        dependencies = []
        
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                
                # Extract tokens
                tokens = [token.text for token in doc]
                
                # Extract lemmas
                lemmatized_tokens = [token.lemma_ for token in doc]
                
                # Extract POS tags
                pos_tags = [(token.text, token.pos_) for token in doc]
                
                # Extract dependencies
                dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            except Exception as e:
                self.logger.error(f"Error parsing with spaCy: {e}")
        else:
            # Fallback: basic tokenization
            tokens = text.split()
            lemmatized_tokens = tokens  # No lemmatization possible without spaCy
            pos_tags = [(token, "UNKNOWN") for token in tokens]
            dependencies = []
        
        # Identify intents
        intents = self.identify_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        return ParsedUtterance(
            original_text=text,
            intents=intents,
            entities=entities,
            tokens=tokens,
            lemmatized_tokens=lemmatized_tokens,
            pos_tags=pos_tags,
            dependencies=dependencies,
            timestamp=time.time()
        )
    
    def process_instruction(self, instruction: str) -> Dict[str, Any]:
        """Process a natural language instruction and return structured interpretation."""
        self.logger.info(f"Processing instruction: '{instruction}'")
        
        # Parse the utterance
        parsed = self.parse_utterance(instruction)
        
        # Select the highest confidence intent
        if parsed.intents:
            primary_intent = parsed.intents[0]
            
            # Format the result
            result = {
                "status": "success",
                "primary_intent": {
                    "name": primary_intent.name,
                    "confidence": primary_intent.confidence,
                    "parameters": primary_intent.parameters
                },
                "all_intents": [
                    {
                        "name": intent.name,
                        "confidence": intent.confidence,
                        "parameters": intent.parameters
                    } for intent in parsed.intents
                ],
                "entities": parsed.entities,
                "tokens": parsed.tokens,
                "original_text": parsed.original_text,
                "timestamp": parsed.timestamp
            }
            
            self.logger.info(f"Recognized intent: {primary_intent.name} (confidence: {primary_intent.confidence:.2f})")
        else:
            result = {
                "status": "error",
                "error": "Could not parse the instruction",
                "original_text": instruction
            }
        
        return result


class DialogStateTracker:
    """Tracks the state of a dialog between human and robot."""
    
    def __init__(self):
        self.conversation_history: List[ParsedUtterance] = []
        self.current_intent: Optional[Intent] = None
        self.entities_memory: Dict[str, List[Any]] = {}
        self.context_variables: Dict[str, Any] = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the dialog state tracker."""
        logger = logging.getLogger("DialogStateTracker")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def update_with_utterance(self, parsed_utterance: ParsedUtterance):
        """Update dialog state with a new utterance."""
        self.conversation_history.append(parsed_utterance)
        
        # Update current intent if there are intents
        if parsed_utterance.intents:
            self.current_intent = parsed_utterance.intents[0]
        
        # Update entities memory
        for entity in parsed_utterance.entities:
            entity_type = entity.get("label", "unknown")
            if entity_type not in self.entities_memory:
                self.entities_memory[entity_type] = []
            self.entities_memory[entity_type].append(entity)
        
        self.logger.info(f"Updated dialog state with: {parsed_utterance.original_text}")
    
    def get_recent_entities(self, entity_type: str) -> List[Any]:
        """Get recently mentioned entities of a specific type."""
        return self.entities_memory.get(entity_type, [])
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current dialog context."""
        return {
            "conversation_length": len(self.conversation_history),
            "current_intent": self.current_intent.name if self.current_intent else None,
            "current_intent_confidence": self.current_intent.confidence if self.current_intent else None,
            "recent_entities": self.entities_memory,
            "context_variables": self.context_variables
        }
    
    def set_context_variable(self, key: str, value: Any):
        """Set a context variable."""
        self.context_variables[key] = value
    
    def get_context_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context_variables.get(key, default)


class SpokenLanguageUnderstanding:
    """Higher-level interface for spoken language understanding in robotics."""
    
    def __init__(self):
        self.nlu_pipeline = NLUPipeline()
        self.dialog_tracker = DialogStateTracker()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the SLU component."""
        logger = logging.getLogger("SpokenLanguageUnderstanding")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def understand_speech(self, speech_text: str) -> Dict[str, Any]:
        """Process speech text and return understanding results."""
        self.logger.info(f"Understanding speech: '{speech_text}'")
        
        # Process the speech with NLU pipeline
        interpretation = self.nlu_pipeline.process_instruction(speech_text)
        
        # If successful, update dialog state
        if interpretation["status"] == "success":
            parsed_utterance = self.nlu_pipeline.parse_utterance(speech_text)
            self.dialog_tracker.update_with_utterance(parsed_utterance)
        
        # Add dialog context to the result
        interpretation["dialog_context"] = self.dialog_tracker.get_context()
        
        self.logger.info(f"Speech understanding result: {interpretation['status']}")
        return interpretation
    
    def handle_follow_up(self, follow_up_text: str) -> Dict[str, Any]:
        """Handle a follow-up question that may reference previous context."""
        # Check if this is a follow-up question (contains pronouns like 'it', 'that', 'the object')
        if any(pronoun in follow_up_text.lower() for pronoun in ['it', 'that', 'the object', 'this']):
            # Try to resolve references using dialog state
            context = self.dialog_tracker.get_context()
            if context.get("current_intent"):
                # Modify the follow-up to include context
                current_obj = context["current_intent"]["parameters"].get("object", "")
                if current_obj:
                    resolved_text = follow_up_text.replace("it", current_obj).replace("that", current_obj)
                    return self.understand_speech(resolved_text)
        
        # Just process as normal if not a contextual follow-up
        return self.understand_speech(follow_up_text)
    
    def get_interpretation_summary(self, interpretation: Dict[str, Any]) -> str:
        """Get a textual summary of the interpretation."""
        if interpretation["status"] != "success":
            return f"Failed to understand: {interpretation.get('error', 'Unknown error')}"
        
        primary_intent = interpretation["primary_intent"]
        entities = interpretation["entities"]
        obj = primary_intent["parameters"].get("object", "something")
        loc = primary_intent["parameters"].get("location", "somewhere")
        
        if primary_intent["name"] == "move_to_location":
            return f"I understood you want me to move to {loc}."
        elif primary_intent["name"] == "grasp_object":
            return f"I understood you want me to grasp the {obj}."
        elif primary_intent["name"] == "navigate_to_object":
            return f"I understood you want me to navigate to the {obj}."
        elif primary_intent["name"] == "general_query":
            return f"I heard you say '{interpretation['original_text']}' but I'm not sure what to do with that."
        else:
            return f"I understood a '{primary_intent['name']}' action with confidence {primary_intent['confidence']:.2f}."


def example_usage():
    """Example of how to use the Natural Language Understanding system."""
    print("Natural Language Understanding Example")
    print("=" * 40)
    
    # Create the NLU system
    slu = SpokenLanguageUnderstanding()
    
    # Test various instructions
    test_instructions = [
        "Please pick up the red cup on the table",
        "Move to the kitchen",
        "Navigate to the large plant near the window",
        "Follow the person with the blue shirt",
        "Go to the door",
        "Grasp the book on the shelf"
    ]
    
    for instruction in test_instructions:
        print(f"\nProcessing: '{instruction}'")
        result = slu.understand_speech(instruction)
        
        if result["status"] == "success":
            primary = result["primary_intent"]
            print(f"  Intent: {primary['name']} (confidence: {primary['confidence']:.2f})")
            print(f"  Parameters: {primary['parameters']}")
            print(f"  Summary: {slu.get_interpretation_summary(result)}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Test dialog context
    print(f"\nTesting dialog context...")
    result1 = slu.understand_speech("Can you pick up the book?")
    print(f"Initial understanding: {slu.get_interpretation_summary(result1)}")
    
    result2 = slu.handle_follow_up("Now put it on the table")
    print(f"Follow-up understanding: {slu.get_interpretation_summary(result2)}")
    
    # Show dialog context
    context = slu.dialog_tracker.get_context()
    print(f"\nDialog context:")
    print(f"  Conversation length: {context['conversation_length']}")
    print(f"  Current intent: {context['current_intent']}")
    print(f"  Current intent confidence: {context['current_intent_confidence']}")


if __name__ == "__main__":
    example_usage()