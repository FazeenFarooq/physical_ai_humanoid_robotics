"""
ROS 2 Debugger and Diagnostic Tools for the Physical AI & Humanoid Robotics Course.
This module provides tools for debugging and diagnosing ROS 2 communication issues.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterEvent
from std_msgs.msg import String, Bool, Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import sys
import time
from collections import defaultdict, deque
import threading


class ROS2Debugger(Node):
    """
    A comprehensive debugging tool for ROS 2 communication.
    Provides functionality to monitor, diagnose, and debug ROS 2 systems.
    """
    
    def __init__(self):
        super().__init__('ros2_debugger')
        
        # Data structures for monitoring
        self.topic_stats = defaultdict(lambda: {'count': 0, 'last_msg_time': 0, 'history': deque(maxlen=100)})
        self.active_topics = set()
        self.subscribers = {}
        self.publishers = {}
        
        # Create a timer to periodically check system status
        self.status_timer = self.create_timer(1.0, self.check_system_status)
        self.message_monitor_timer = self.create_timer(0.1, self.monitor_messages)
        
        # Publishers for debug information
        self.debug_publisher = self.create_publisher(String, 'debug_info', 10)
        self.status_publisher = self.create_publisher(String, 'system_status', 10)
        
        # Parameter event subscription for parameter changes
        self.parameter_sub = self.create_subscription(
            ParameterEvent,
            '/parameter_events',
            self.parameter_event_callback,
            10
        )
        
        self.get_logger().info('ROS 2 Debugger initialized')
    
    def parameter_event_callback(self, msg):
        """Handle parameter change events."""
        for changed_parameter in msg.changed_parameters:
            self.get_logger().info(
                f'Parameter {changed_parameter.name} changed to {changed_parameter.value}'
            )
    
    def monitor_messages(self):
        """Monitor message publishing and subscription."""
        current_time = time.time()
        
        # Log periodic status update
        status_msg = String()
        status_msg.data = f"Active topics: {len(self.active_topics)}, " \
                         f"Messages processed: {sum(s['count'] for s in self.topic_stats.values())}"
        self.status_publisher.publish(status_msg)
    
    def check_system_status(self):
        """Check the overall system status."""
        # This would check for common issues like:
        # - Unresponsive nodes
        # - Message delays
        # - Parameter mismatches
        # - Resource constraints
        
        self.get_logger().info(f"System Status - Active topics: {len(self.active_topics)}")
        
        # Check for potentially problematic topics (no messages for a while)
        current_time = time.time()
        for topic, stats in self.topic_stats.items():
            time_since_last = current_time - stats['last_msg_time']
            if time_since_last > 5.0:  # If no message for more than 5 seconds
                self.get_logger().warning(f"Topic {topic} has not received messages for {time_since_last:.2f}s")
    
    def add_topic_monitor(self, topic_name, msg_type):
        """Add a topic to be monitored."""
        if topic_name not in self.subscribers:
            # Create subscriber for the topic
            subscriber = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, topic=topic_name: self.message_callback(msg, topic),
                10
            )
            self.subscribers[topic_name] = subscriber
            self.active_topics.add(topic_name)
            self.get_logger().info(f"Added monitor for topic: {topic_name}")
    
    def message_callback(self, msg, topic_name):
        """Callback for monitored messages."""
        current_time = time.time()
        
        # Update statistics
        self.topic_stats[topic_name]['count'] += 1
        self.topic_stats[topic_name]['last_msg_time'] = current_time
        self.topic_stats[topic_name]['history'].append({
            'timestamp': current_time,
            'message': str(msg)[:100]  # Truncate long messages
        })
    
    def get_topic_info(self, topic_name):
        """Get information about a specific topic."""
        if topic_name in self.topic_stats:
            stats = self.topic_stats[topic_name]
            return {
                'topic_name': topic_name,
                'message_count': stats['count'],
                'last_message_time': stats['last_msg_time'],
                'message_history': list(stats['history'])[-10:]  # Last 10 messages
            }
        return None
    
    def get_all_topics_info(self):
        """Get information about all monitored topics."""
        return {
            topic_name: {
                'message_count': stats['count'],
                'last_message_time': stats['last_msg_time'],
                'recent_messages': list(stats['history'])[-5:]  # Last 5 messages
            }
            for topic_name, stats in self.topic_stats.items()
        }
    
    def diagnose_connection_issues(self):
        """Diagnose common connection issues."""
        issues = []
        
        # Check if we're receiving messages on expected topics
        expected_topics = [  # These would be configurable in a real system
            '/cmd_vel',
            '/joint_states',
            '/tf',
            '/tf_static'
        ]
        
        for topic in expected_topics:
            if topic in self.topic_stats:
                time_since_last = time.time() - self.topic_stats[topic]['last_msg_time']
                if time_since_last > 2.0:
                    issues.append(f"Expected topic {topic} has not received messages recently")
            else:
                issues.append(f"Expected topic {topic} is not active")
        
        return issues
    
    def print_detailed_status(self):
        """Print detailed system status."""
        self.get_logger().info("=== ROS 2 System Status ===")
        self.get_logger().info(f"Active topics: {len(self.active_topics)}")
        self.get_logger().info(f"Total messages processed: {sum(s['count'] for s in self.topic_stats.values())}")
        
        for topic, stats in self.topic_stats.items():
            self.get_logger().info(f"  {topic}: {stats['count']} messages")
        
        # Diagnose issues
        issues = self.diagnose_connection_issues()
        if issues:
            self.get_logger().info("Potential issues detected:")
            for issue in issues:
                self.get_logger().warning(f"  {issue}")
        else:
            self.get_logger().info("No immediate issues detected")


class TopicMonitor(Node):
    """
    A utility node for monitoring specific topics and providing detailed information.
    """
    
    def __init__(self, topics_to_monitor):
        super().__init__('topic_monitor')
        
        self.topics_to_monitor = topics_to_monitor
        self.message_counts = defaultdict(int)
        self.subscribers = {}
        
        # Create subscribers for each topic
        for topic_name, msg_type in topics_to_monitor:
            self.subscribers[topic_name] = self.create_subscription(
                msg_type,
                topic_name,
                self.create_callback(topic_name),
                10
            )
            self.get_logger().info(f"Monitoring topic: {topic_name}")
    
    def create_callback(self, topic_name):
        """Create a callback function for a specific topic."""
        def callback(msg):
            self.message_counts[topic_name] += 1
            # For this example, just log the message count periodically
            if self.message_counts[topic_name] % 10 == 0:
                self.get_logger().info(f"{topic_name}: {self.message_counts[topic_name]} messages")
        return callback
    
    def get_stats(self):
        """Get statistics for monitored topics."""
        return dict(self.message_counts)


class MessageValidator(Node):
    """
    A utility to validate message contents against expected schemas/values.
    """
    
    def __init__(self):
        super().__init__('message_validator')
        self.validators = {}
        self.errors = defaultdict(list)
    
    def add_validator(self, topic_name, validation_func):
        """Add a validation function for a specific topic."""
        self.validators[topic_name] = validation_func
    
    def validate_message(self, topic_name, msg):
        """Validate a message using the appropriate validator."""
        if topic_name in self.validators:
            try:
                is_valid, error_msg = self.validators[topic_name](msg)
                if not is_valid:
                    self.errors[topic_name].append({
                        'timestamp': time.time(),
                        'error': error_msg,
                        'message': str(msg)
                    })
                    self.get_logger().error(f"Validation error on {topic_name}: {error_msg}")
            except Exception as e:
                self.errors[topic_name].append({
                    'timestamp': time.time(),
                    'error': f"Validator exception: {str(e)}",
                    'message': str(msg)
                })
                self.get_logger().error(f"Validator exception on {topic_name}: {str(e)}")
    
    def get_errors(self, topic_name=None):
        """Get validation errors, optionally filtered by topic."""
        if topic_name:
            return self.errors[topic_name]
        return dict(self.errors)


def main(args=None):
    """Main function to run the ROS 2 debugger."""
    rclpy.init(args=args)
    
    debugger = ROS2Debugger()
    
    # Example: Add some common topics to monitor
    debugger.add_topic_monitor('/cmd_vel', Twist)
    debugger.add_topic_monitor('/joint_states', JointState)
    
    try:
        debugger.get_logger().info("ROS 2 Debugger running. Press Ctrl+C to stop.")
        debugger.print_detailed_status()  # Print initial status
        
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info("Shutting down ROS 2 Debugger")
    finally:
        debugger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()