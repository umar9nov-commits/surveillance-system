from setuptools import find_packages, setup

package_name = 'custom_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='umaryasir',
    maintainer_email='umaryasir@todo.todo',
    description='Custom A* planning and robot navigation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'planner_node = custom_navigation.planner_node:main',
            'tb3_mpc_controller = custom_navigation.tb3_mpc_controller:main',
        ],
    },
)
