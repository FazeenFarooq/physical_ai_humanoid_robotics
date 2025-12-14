from setuptools import setup

package_name = 'robot_communication'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Physical AI Course Maintainer',
    maintainer_email='maintainer@physicalai.edu',
    description='ROS 2 communication framework for Physical AI & Humanoid Robotics course',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = robot_communication.talker:main',
            'listener = robot_communication.listener:main',
        ],
    },
)