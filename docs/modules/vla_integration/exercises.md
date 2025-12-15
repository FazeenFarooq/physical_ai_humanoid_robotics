# Vision-Language-Action (VLA) Integration Exercises

## Overview

This module covers the integration of Vision, Language, and Action systems to create unified cognitive robotic systems. Students will learn to implement systems that can understand natural language commands, perceive the environment, and execute appropriate actions to achieve user goals.

## Learning Objectives

- Understand the architecture of Vision-Language-Action systems
- Implement multimodal perception pipelines
- Integrate natural language understanding with robotic action execution
- Create cognitive stacks that combine perception, reasoning, and action
- Evaluate the performance of integrated VLA systems

## Exercise 1: Setting up the VLA Architecture

### Objective
Implement the foundational architecture for Vision-Language-Action integration.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Basic understanding of perception and action systems
- Python 3.10+ with PyTorch 2.0+

### Steps
1. Create the VLA architecture components:
   ```bash
   mkdir -p ~/ai_robotics_ws/src/vla_integration/src
   touch ~/ai_robotics_ws/src/vla_integration/src/__init__.py
   ```

2. Create the VLA main module:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/vla_manager.py
   import rospy
   import torch
   from sensor_msgs.msg import Image
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   
   class VLAIntegrationManager:
       def __init__(self):
           rospy.init_node('vla_integration_manager')
           
           # Publishers and subscribers
           self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
           self.command_sub = rospy.Subscriber('/vla/command', String, self.command_callback)
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           
           # Initialize perception and language components
           self.perception_system = PerceptionSystem()
           self.language_system = LanguageUnderstandingSystem()
           self.action_system = ActionExecutionSystem()
           
       def image_callback(self, data):
           # Process incoming image data
           image = self.process_image(data)
           self.perception_system.process(image)
           
       def command_callback(self, data):
           # Process incoming natural language command
           command = data.data
           intent = self.language_system.understand(command)
           action = self.action_system.plan_action(intent, self.perception_system.get_environment_state())
           self.execute_action(action)
           
       def execute_action(self, action):
           # Execute the planned action
           self.action_system.execute(action)
           
       def run(self):
           rospy.spin()
   ```

3. Create the perception system:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/perception.py
   import torch
   import cv2
   import numpy as np
   from transformers import DetrImageProcessor, DetrForObjectDetection
   
   class PerceptionSystem:
       def __init__(self):
           # Initialize object detection model
           self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
           self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
           
       def process(self, image):
           inputs = self.processor(images=image, return_tensors="pt")
           outputs = self.model(**inputs)
           target_sizes = torch.tensor([image.shape[:2]])
           results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
           
           objects = []
           for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
               objects.append({
                   "label": self.model.config.id2label[label.item()],
                   "confidence": score.item(),
                   "bbox": box.tolist()
               })
               
           return objects
           
       def get_environment_state(self):
           # Return current environment state based on latest perception
           pass
   ```

4. Create the language understanding system:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/language_understanding.py
   from transformers import pipeline
   
   class LanguageUnderstandingSystem:
       def __init__(self):
           self.nlu_pipeline = pipeline("text-classification", model="microsoft/DialoGPT-medium")
           
       def understand(self, command):
           # Parse the natural language command to extract intent and entities
           result = self.nlu_pipeline(command)
           return self.parse_command(result, command)
           
       def parse_command(self, classification_result, command):
           # Extract intent and entities from the command
           intent = classification_result['label']
           return {
               "intent": intent,
               "command": command,
               "entities": self.extract_entities(command)
           }
           
       def extract_entities(self, command):
           # Extract named entities from the command
           # This could use regex patterns, NER, or other techniques
           return []
   ```

5. Create the action execution system:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/action_execution.py
   from geometry_msgs.msg import Twist
   import rospy
   
   class ActionExecutionSystem:
       def __init__(self):
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           
       def plan_action(self, intent, environment_state):
           # Plan action based on intent and environment state
           if intent["intent"] == "move_forward":
               return {"action_type": "navigation", "params": {"linear_x": 0.5, "angular_z": 0.0}}
           elif intent["intent"] == "turn_left":
               return {"action_type": "navigation", "params": {"linear_x": 0.0, "angular_z": 0.5}}
           # Add more action planning logic as needed
           
           return {"action_type": "none", "params": {}}
           
       def execute(self, action):
           # Execute the planned action
           if action["action_type"] == "navigation":
               cmd = Twist()
               cmd.linear.x = action["params"]["linear_x"]
               cmd.angular.z = action["params"]["angular_z"]
               self.cmd_vel_pub.publish(cmd)
   ```

## Exercise 2: Implementing a Simple VLA Task

### Objective
Create a complete VLA pipeline that can execute a simple navigation task based on natural language.

### Prerequisites
- VLA architecture from Exercise 1
- Camera sensor in simulation or on physical robot

### Steps
1. Create a launch file for the VLA system:
   ```xml
   <!-- ~/ai_robotics_ws/src/vla_integration/launch/vla_system.launch.py -->
   from launch import LaunchDescription
   from launch_ros.actions import Node
   
   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='vla_integration',
               executable='vla_manager',
               name='vla_integration_manager',
               output='screen'
           )
       ])
   ```

2. Create a simple test script:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/scripts/test_vla.py
   import rospy
   from std_msgs.msg import String
   
   def test_vla():
       rospy.init_node('vla_tester', anonymous=True)
       pub = rospy.Publisher('/vla/command', String, queue_size=10)
       rospy.sleep(1)  # Wait for publisher to connect
       
       # Send commands to the VLA system
       commands = [
           "move forward",
           "turn left",
           "stop"
       ]
       
       for cmd in commands:
           rospy.loginfo(f"Sending command: {cmd}")
           pub.publish(String(data=cmd))
           rospy.sleep(3)  # Wait 3 seconds between commands
           
   if __name__ == '__main__':
       try:
           test_vla()
       except rospy.ROSInterruptException:
           pass
   ```

3. Test the VLA system in simulation:
   ```bash
   # Terminal 1: Launch the robot simulation
   ros2 launch turtlebot4_gz_bringup turtlebot4_ignition.launch.py
   
   # Terminal 2: Launch the VLA system
   ros2 launch vla_integration vla_system.launch.py
   
   # Terminal 3: Send test commands
   ros2 run vla_integration test_vla.py
   ```

## Exercise 3: Advanced VLA with Object Manipulation

### Objective
Extend the VLA system to include object manipulation based on natural language commands.

### Prerequisites
- Basic VLA system from previous exercises
- Manipulation capabilities in the robot (arm, gripper)

### Steps
1. Extend the perception system to identify graspable objects:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/perception.py (extension)
   class PerceptionSystem:
       # ... existing code ...
       
       def identify_graspable_objects(self, image):
           # Identify objects that can be grasped by the robot
           objects = self.process(image)
           graspable = []
           
           for obj in objects:
               if obj["label"] in ["bottle", "cup", "box", "can"]:
                   graspable.append(obj)
                   
           return graspable
   ```

2. Extend the language system to understand manipulation commands:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/language_understanding.py (extension)
   class LanguageUnderstandingSystem:
       # ... existing code ...
       
       def parse_command(self, classification_result, command):
           # Enhanced parsing for manipulation commands
           intent = classification_result['label']
           
           if "pick up" in command or "grasp" in command:
               intent = "grasp_object"
           elif "move" in command and ("to" in command or "toward" in command):
               intent = "navigate_to_object"
               
           return {
               "intent": intent,
               "command": command,
               "entities": self.extract_entities(command),
               "target_object": self.extract_target_object(command)
           }
           
       def extract_target_object(self, command):
           # Extract the target object from the command
           for obj in ["bottle", "cup", "box", "can", "ball"]:
               if obj in command:
                   return obj
           return None
   ```

3. Add manipulation capabilities to the action system:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/src/action_execution.py (extension)
   class ActionExecutionSystem:
       def __init__(self):
           # ... existing initialization ...
           # Add manipulation publishers if available
           
       def plan_action(self, intent, environment_state):
           # ... existing code ...
           
           if intent["intent"] == "grasp_object":
               target_obj = intent.get("target_object")
               if target_obj:
                   # Plan manipulation action
                   return {"action_type": "manipulation", 
                          "params": {"action": "grasp", "target": target_obj}}
           
           return super().plan_action(intent, environment_state)
   ```

## Exercise 4: VLA System Evaluation and Tuning

### Objective
Evaluate the performance of the VLA system and tune parameters for better accuracy.

### Prerequisites
- Working VLA system from previous exercises
- Test commands and expected outcomes

### Steps
1. Create an evaluation script:
   ```python
   # ~/ai_robotics_ws/src/vla_integration/scripts/evaluate_vla.py
   import rospy
   import time
   from std_msgs.msg import String
   from std_msgs.msg import Bool
   
   class VLAEvaluator:
       def __init__(self):
           rospy.init_node('vla_evaluator')
           self.command_pub = rospy.Publisher('/vla/command', String, queue_size=10)
           self.success_sub = rospy.Subscriber('/vla/success', Bool, self.success_callback)
           
           self.tests = [
               {"command": "move forward", "expected": "move_forward", "timeout": 5.0},
               {"command": "turn left", "expected": "turn_left", "timeout": 5.0},
               {"command": "stop", "expected": "stop", "timeout": 5.0}
           ]
           
           self.current_test_idx = 0
           self.test_results = []
           
       def success_callback(self, msg):
           # Record result of the current test
           if self.current_test_idx < len(self.tests):
               self.test_results.append(msg.data)
               
       def run_evaluation(self):
           for i, test in enumerate(self.tests):
               print(f"Running test {i+1}: {test['command']}")
               self.command_pub.publish(String(data=test['command']))
               time.sleep(test['timeout'])
               
               # Check if we have a result for this test
               if len(self.test_results) > i:
                   success = self.test_results[i]
                   print(f"Test {i+1} result: {'PASS' if success else 'FAIL'}")
               else:
                   print(f"Test {i+1} result: TIMEOUT")
                   self.test_results.append(False)
                   
           # Calculate overall accuracy
           accuracy = sum(self.test_results) / len(self.test_results)
           print(f"Overall accuracy: {accuracy * 100:.2f}%")
           
   if __name__ == '__main__':
       evaluator = VLAEvaluator()
       rospy.sleep(2)  # Wait for connections
       evaluator.run_evaluation()
   ```

2. Tune system parameters based on evaluation results:
   - Adjust perception thresholds
   - Fine-tune language understanding models
   - Optimize action execution timing

## Assessment

Students will be assessed on:
1. Successful implementation of the VLA architecture
2. Ability to execute simple navigation tasks from language commands
3. Extension of the system to include manipulation
4. Performance evaluation and parameter tuning
5. Integration quality between vision, language, and action components

## References

- [Vision-Language-Action Models Research](https://arxiv.org/abs/2206.00451)
- [Robotic Transformers for Vision-Language-Action Tasks](https://ai.googleblog.com/2022/02/rt1-robotics-transformer-for-real-world.html)
- [ROS 2 Navigation and Manipulation Tutorials](https://navigation.ros.org/)