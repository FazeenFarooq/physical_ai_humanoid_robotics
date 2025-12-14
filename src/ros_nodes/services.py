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
ROS 2 Service Examples for the Physical AI & Humanoid Robotics Course.
This module demonstrates how to create and use ROS 2 services for request-response communication.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from std_srvs.srv import SetBool
from geometry_msgs.msg import Point
import math


class SimpleArithmeticService(Node):
    """
    A simple service server that performs arithmetic operations.
    Demonstrates basic service server implementation in ROS 2.
    """
    
    def __init__(self):
        super().__init__('simple_arithmetic_service')
        
        # Create a service that adds two integers
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )
        
        self.get_logger().info('Simple arithmetic service server started')
    
    def add_two_ints_callback(self, request, response):
        """Callback function for the add_two_ints service."""
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request: {request.a} + {request.b} = {response.sum}')
        return response


class SimpleArithmeticClient(Node):
    """
    A client for the simple arithmetic service.
    Demonstrates how to call a ROS 2 service from a client node.
    """
    
    def __init__(self):
        super().__init__('simple_arithmetic_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = AddTwoInts.Request()
    
    def send_request(self, a, b):
        """Send a request to the service."""
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        return self.future


class RobotControlService(Node):
    """
    A service server for controlling a robot.
    Demonstrates service usage for robot control commands.
    """
    
    def __init__(self):
        super().__init__('robot_control_service')
        
        # Service to set robot velocity
        self.velocity_srv = self.create_service(
            SetBool,
            'set_robot_velocity',
            self.set_robot_velocity_callback
        )
        
        # Service to calculate distance to target
        self.distance_srv = self.create_service(
            AddTwoInts,  # Using AddTwoInts as a simple example, would use custom message in practice
            'calculate_distance',
            self.calculate_distance_callback
        )
        
        self.get_logger().info('Robot control service server started')
    
    def set_robot_velocity_callback(self, request, response):
        """Handle request to set robot velocity."""
        if request.data:
            self.get_logger().info('Setting robot to move forward')
            # In a real robot, this would send commands to the motion controller
        else:
            self.get_logger().info('Stopping robot')
            # In a real robot, this would stop the motion controller
        
        response.success = True
        response.message = "Velocity command processed"
        return response
    
    def calculate_distance_callback(self, request, response):
        """Calculate distance between two points (simplified example)."""
        # In a real implementation, this would take two points as input
        # Here using request.a and request.b as x and y coordinates of a point
        # with the origin at (0, 0)
        distance = math.sqrt(request.a**2 + request.b**2)
        response.sum = int(distance)
        self.get_logger().info(f'Distance from origin to point ({request.a}, {request.b}) is {distance}')
        return response


class RobotControlClient(Node):
    """
    A client for the robot control service.
    Demonstrates how to use the robot control services.
    """
    
    def __init__(self):
        super().__init__('robot_control_client')
        
        # Client for setting robot velocity
        self.velocity_cli = self.create_client(SetBool, 'set_robot_velocity')
        
        # Client for calculating distance
        self.distance_cli = self.create_client(AddTwoInts, 'calculate_distance')
        
        # Wait for services to be available
        while not self.velocity_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Velocity service not available, waiting again...')
        
        while not self.distance_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Distance service not available, waiting again...')
        
        self.velocity_req = SetBool.Request()
        self.distance_req = AddTwoInts.Request()
    
    def set_robot_velocity(self, forward):
        """Request to set robot velocity."""
        self.velocity_req.data = forward
        self.future = self.velocity_cli.call_async(self.velocity_req)
        return self.future
    
    def calculate_distance(self, x, y):
        """Request to calculate distance to a point."""
        self.distance_req.a = int(x * 100)  # Converting to int for example
        self.distance_req.b = int(y * 100)  # Converting to int for example
        self.future = self.distance_cli.call_async(self.distance_req)
        return self.future


def main_arithmetic(args=None):
    """Main function to demonstrate arithmetic service."""
    rclpy.init(args=args)
    
    service_node = SimpleArithmeticService()
    client_node = SimpleArithmeticClient()
    
    # Send a request from the client
    future = client_node.send_request(2, 3)
    
    try:
        rclpy.spin_until_future_complete(client_node, future)
        response = future.result()
        client_node.get_logger().info(f'Result of 2 + 3: {response.sum}')
    except KeyboardInterrupt:
        service_node.get_logger().info('Interrupted during service call')
    finally:
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


def main_robot_control(args=None):
    """Main function to demonstrate robot control service."""
    rclpy.init(args=args)
    
    service_node = RobotControlService()
    client_node = RobotControlClient()
    
    try:
        # Example of calling the services
        # First, set robot to move
        future1 = client_node.set_robot_velocity(True)
        rclpy.spin_until_future_complete(client_node, future1)
        result1 = future1.result()
        client_node.get_logger().info(f'Velocity command success: {result1.success}')
        
        # Then, calculate distance to a point
        future2 = client_node.calculate_distance(3.0, 4.0)
        rclpy.spin_until_future_complete(client_node, future2)
        result2 = future2.result()
        client_node.get_logger().info(f'Distance calculated: {result2.sum/100.0} units')  # Converting back from int
        
    except KeyboardInterrupt:
        service_node.get_logger().info('Interrupted during service calls')
    finally:
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # To run the arithmetic example:
    # main_arithmetic()
    
    # To run the robot control example:
    main_robot_control()