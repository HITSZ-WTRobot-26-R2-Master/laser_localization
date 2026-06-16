cmake_minimum_required(VERSION 3.8)
project(agv_pose_refiner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(interfaces REQUIRED)

ament_python_install_package(agv_pose_refiner_py)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_common test/test_common.py)
  ament_add_pytest_test(test_pose_solver test/test_pose_solver.py)
endif()

install(PROGRAMS
  scripts/agv_pose_refiner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(interfaces)

ament_package()
