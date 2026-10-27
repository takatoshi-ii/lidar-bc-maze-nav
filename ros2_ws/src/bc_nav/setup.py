from glob import glob
from setuptools import setup

package_name = 'bc_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Behavior Cloning for LiDAR navigation',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'bc_teleop = bc_nav.bc_teleop:main',
            'data_recorder = bc_nav.data_recorder:main',
            'train_bc = bc_nav.train_bc:main',
            'bc_driver = bc_nav.bc_driver:main',
            'video_recorder = bc_nav.video_recorder:main',
        ],
    },
)
