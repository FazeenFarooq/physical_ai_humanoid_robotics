# Copyright 2023 Physical AI & Humanoid Robotics Course
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific purposes governing permissions and
# limitations under the License.

"""
ROS 2 Action Examples for the Physical AI & Humanoid Robotics Course.
This module demonstrates how to create and use ROS 2 actions for goal-oriented tasks.
"""

import rclpy
from rclpy.action import ActionServer, ActionClient
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import action messages (for this example we'll use a built-in action)
from rcl_interfaces.action import SetParameters
import time
import math
from rclpy.action import CancelResponse, GoalResponse
from std_msgs.msg import String


# Since we can't import custom action messages in this context, 
# we'll create a simple class to simulate an action interface
class NavigateToPoseGoal:
    def __init__(self):
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0

class NavigateToPoseResult:
    def __init__(self):
        self.success = False
        self.message = ""

class NavigateToPoseFeedback:
    def __init__(self):
        self.distance_remaining = 0.0
        self.current_speed = 0.0
        self.status = ""


class NavigateToPoseAction:
    """Simulated action interface for navigation."""
    Goal = NavigateToPoseGoal
    Result = NavigateToPoseResult
    Feedback = NavigateToPoseFeedback


class NavigationActionServer(Node):
    """
    Action server that simulates navigating to a pose.
    Demonstrates how to implement a ROS 2 action server.
    """
    
    def __init__(self):
        super().__init__('navigation_action_server')
        
        # Create action server with a reentrant callback group to handle
        # multiple goals simultaneously
        self._action_server = ActionServer(
            self,
            NavigateToPoseAction,
            'navigate_to_pose',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        self.get_logger().info('Navigation action server started')
    
    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info('Received goal request')
        
        # Validate the goal
        if goal_request.target_x > 100 or goal_request.target_y > 100 or goal_request.target_x < -100 or goal_request.target_y < -100:
            self.get_logger().info('Goal rejected - out of bounds')
            return GoalResponse.REJECT
        else:
            self.get_logger().info('Goal accepted')
            return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        # Get the target pose from the goal
        target_x = goal_handle.request.target_x
        target_y = goal_handle.request.target_y
        
        # Start executing the action
        feedback_msg = NavigateToPoseFeedback()
        
        # Simulate navigation by updating feedback periodically
        current_x, current_y = 0.0, 0.0  # Starting position
        
        while rclpy.ok():
            # Check if the goal has been canceled
            if goal_handle.is_canceling():
                result = NavigateToPoseResult()
                result.success = False
                result.message = "Goal was canceled"
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result
            
            # Calculate distance to target
            dist_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # Update feedback
            feedback_msg.distance_remaining = dist_to_target
            feedback_msg.current_speed = 0.5  # Simulated speed
            feedback_msg.status = f"Moving to ({target_x:.2f}, {target_y:.2f})"
            
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f'Distance to goal: {dist_to_target:.2f}, Status: {feedback_msg.status}')
            
            # Update current position (simulating movement)
            if dist_to_target > 0.1:  # If not close enough to target
                # Move toward target (simplified)
                direction_x = (target_x - current_x) / dist_to_target
                direction_y = (target_y - current_y) / dist_to_target
                current_x += direction_x * 0.05  # Move 0.05 units per iteration
                current_y += direction_y * 0.05
                
                time.sleep(0.1)  # Simulate time delay
            else:
                # Reached the target
                result = NavigateToPoseResult()
                result.success = True
                result.message = f"Successfully navigated to ({target_x:.2f}, {target_y:.2f})"
                
                goal_handle.succeed()
                self.get_logger().info(f'Goal succeeded: {result.message}')
                return result


class NavigationActionClient(Node):
    """
    Action client for navigation tasks.
    Demonstrates how to use ROS 2 actions from a client node.
    """
    
    def __init__(self):
        super().__init__('navigation_action_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            NavigateToPoseAction,
            'navigate_to_pose'
        )
    
    def send_goal(self, x, y):
        """Send a navigation goal to the action server."""
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        # Create the goal
        goal_msg = NavigateToPoseGoal()
        goal_msg.target_x = float(x)
        goal_msg.target_y = float(y)
        goal_msg.target_theta = 0.0  # Not used in this example
        
        # Send the goal
        self.get_logger().info(f'Sending goal to ({x}, {y})')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        # Set up result callback
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return send_goal_future
    
    def goal_response_callback(self, future):
        """Handle response when goal is accepted."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected')
            return
        
        self.get_logger().info('Goal was accepted')
        
        # Get the result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)
    
    def result_callback(self, future):
        """Handle the result of the action."""
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')
    
    def feedback_callback(self, feedback_msg):
        """Handle feedback during action execution."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: {feedback.distance_remaining:.2f}m remaining, '
            f'speed: {feedback.current_speed:.2f}m/s, status: {feedback.status}'
        )


def main_navigation_example(args=None):
    """Main function to demonstrate navigation action."""
    rclpy.init(args=args)
    
    action_server = NavigationActionServer()
    action_client = NavigationActionClient()
    
    # Start the action server in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)
    executor.add_node(action_client)
    
    # Send a goal from the client after a short delay
    goal_sent = False
    def send_goal_later():
        nonlocal goal_sent
        if not goal_sent:
            time.sleep(2)  # Wait for server to start
            action_client.send_goal(5, 5)  # Navigate to (5, 5)
            goal_sent = True
    
    # Schedule the goal to be sent
    timer = action_client.create_timer(0.1, send_goal_later)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted during action execution')
    finally:
        timer.destroy()
        action_server.destroy_node()
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Run the navigation example
    main_navigation_example()