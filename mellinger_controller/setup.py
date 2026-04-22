from setuptools import find_packages, setup

package_name = 'mellinger_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/mellinger_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/mellinger.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='kapiotrow@student.agh.edu.pl',
    description='Mellinger geometric controller for quadrotor UAVs.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'mellinger_controller = mellinger_controller.mellinger_node:main',
        ],
    },
)
