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
Basic publisher and subscriber example for the Physical AI & Humanoid Robotics Course.
This demonstrates fundamental ROS 2 communication patterns.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BasicPublisher(Node):
    """
    Basic publisher node that publishes messages to a topic.
    Demonstrates fundamental concepts of ROS 2 publishers.
    """

    def __init__(self):
        super().__init__('basic_publisher')
        self.publisher_ = self.create_publisher(String, 'basic_topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


class BasicSubscriber(Node):
    """
    Basic subscriber node that listens to messages from a topic.
    Demonstrates fundamental concepts of ROS 2 subscribers.
    """

    def __init__(self):
        super().__init__('basic_subscriber')
        self.subscription = self.create_subscription(
            String,
            'basic_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)

    # Create both publisher and subscriber nodes
    publisher_node = BasicPublisher()
    subscriber_node = BasicSubscriber()

    # Run both nodes
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()