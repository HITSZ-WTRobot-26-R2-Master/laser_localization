from glob import glob
from setuptools import find_packages, setup


package_name = "agv_pose_refiner"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML", "pyserial"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="codex@example.com",
    description="Core 6-beam and LiDAR coarse pose localization refiner.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agv_pose_refiner_node = agv_pose_refiner.agv_pose_refiner_node:main",
            "agv_three_beam_pose_test_node = agv_pose_refiner.agv_three_beam_pose_test_node:main",
        ],
    },
)
