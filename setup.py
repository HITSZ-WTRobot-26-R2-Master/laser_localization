from glob import glob

from setuptools import setup

package_name = 'agv_pose_refiner'
python_package_name = 'agv_pose_refiner_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[python_package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='6-beam laser localization refiner',
    license='Apache-2.0',

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],

    entry_points={
        'console_scripts': [
            'agv_pose_refiner_node = agv_pose_refiner_py.agv_pose_refiner_node:main'
        ],
    },
)
