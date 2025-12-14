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
ROS 2 Communication Node Templates for the Physical AI & Humanoid Robotics Course.
These templates provide a foundation for creating various types of ROS 2 nodes.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time


class NodeTemplate(Node):
    """
    Basic template for a ROS 2 node.
    Provides common functionality and structure for other nodes.
    """
    
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f'{node_name} initialized')
    
    def shutdown(self):
        """Perform cleanup operations before shutdown."""
        self.get_logger().info(f'Shutting down {self.get_name()}')


class PublisherNodeTemplate(NodeTemplate):
    """
    Template for a ROS 2 publisher node.
    Demonstrates how to create and use publishers.
    """
    
    def __init__(self, node_name, topic_name, msg_type=String, queue_size=10):
        super().__init__(node_name)
        self.publisher = self.create_publisher(msg_type, topic_name, queue_size)
        self.counter = 0
    
    def publish_message(self, msg):
        """Publish a message to the topic."""
        self.publisher.publish(msg)
        self.counter += 1
        self.get_logger().info(f'Published message #{self.counter}')
    
    def get_message_count(self):
        """Get the number of messages published."""
        return self.counter


class SubscriberNodeTemplate(NodeTemplate):
    """
    Template for a ROS 2 subscriber node.
    Demonstrates how to create and use subscribers.
    """
    
    def __init__(self, node_name, topic_name, msg_type=String, queue_size=10, callback=None):
        super().__init__(node_name)
        self.subscriber = self.create_subscription(
            msg_type,
            topic_name,
            callback if callback else self.default_callback,
            queue_size)
        self.message_count = 0
    
    def default_callback(self, msg):
        """Default callback function for message reception."""
        self.get_logger().info(f'Received message: {msg}')
        self.message_count += 1
    
    def get_message_count(self):
        """Get the number of messages received."""
        return self.message_count


class RobotControlNodeTemplate(NodeTemplate):
    """
    Template for a robot control node.
    Demonstrates how to control robot motion through Twist messages.
    """
    
    def __init__(self, node_name):
        super().__init__(node_name)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Current desired velocity
        self.linear_vel = 0.0
        self.angular_vel = 0.0
    
    def set_velocity(self, linear, angular):
        """Set the desired linear and angular velocity."""
        self.linear_vel = linear
        self.angular_vel = angular
    
    def stop(self):
        """Stop the robot."""
        self.set_velocity(0.0, 0.0)
    
    def control_loop(self):
        """Publish velocity commands at regular intervals."""
        msg = Twist()
        msg.linear.x = self.linear_vel
        msg.angular.z = self.angular_vel
        self.cmd_vel_publisher.publish(msg)


class JointStateNodeTemplate(NodeTemplate):
    """
    Template for a joint state publisher node.
    Demonstrates how to publish joint positions and states.
    """
    
    def __init__(self, node_name, joint_names):
        super().__init__(node_name)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.joint_names = joint_names
        self.joint_positions = [0.0] * len(joint_names)
        self.joint_velocities = [0.0] * len(joint_names)
        self.joint_efforts = [0.0] * len(joint_names)
        
        # Timer for publishing joint states
        self.timer = self.create_timer(0.05, self.publish_joint_states)
    
    def set_joint_position(self, joint_name, position):
        """Set the position of a specific joint."""
        if joint_name in self.joint_names:
            idx = self.joint_names.index(joint_name)
            self.joint_positions[idx] = position
    
    def set_joint_positions(self, positions):
        """Set positions for all joints."""
        if len(positions) == len(self.joint_names):
            self.joint_positions = list(positions)
    
    def publish_joint_states(self):
        """Publish the current joint states."""
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.joint_state_publisher.publish(msg)


# Example usage of the templates
def example_usage():
    """
    Example of how to use the templates to create specific nodes.
    """
    rclpy.init()
    
    # Create a publisher node
    pub_node = PublisherNodeTemplate('example_publisher', 'chatter', String)
    
    # Create a subscriber node
    def custom_callback(msg):
        print(f'Custom callback received: {msg.data}')
    
    sub_node = SubscriberNodeTemplate('example_subscriber', 'chatter', String, callback=custom_callback)
    
    # Create a robot control node
    control_node = RobotControlNodeTemplate('robot_controller')
    control_node.set_velocity(0.5, 0.0)  # Move forward at 0.5 m/s
    
    # Create a joint state node
    joint_names = ['joint1', 'joint2', 'joint3']
    joint_node = JointStateNodeTemplate('joint_publisher', joint_names)
    joint_node.set_joint_positions([0.1, 0.2, 0.3])  # Set joint positions
    
    # Spin all nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(pub_node)
    executor.add_node(sub_node)
    executor.add_node(control_node)
    executor.add_node(joint_node)
    
    try:
        print('Starting nodes, press Ctrl+C to stop...')
        executor.spin()
    except KeyboardInterrupt:
        print('Shutting down nodes...')
    finally:
        pub_node.shutdown()
        sub_node.shutdown()
        control_node.shutdown()
        joint_node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    example_usage()