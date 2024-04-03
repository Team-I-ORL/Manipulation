cmake_minimum_required(VERSION 3.22)
project(fetch_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_msgs REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)
find_package(urdf REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(orbiter_bt REQUIRED)



install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

set (TARGET_DEPENDS
  orbiter_bt
  rclcpp
  moveit_msgs
  moveit_ros_planning_interface
  urdf
  visualization_msgs
)

add_executable(moveit_interface_wrapper src/moveit_interface_wrapper.cpp)
ament_target_dependencies(moveit_interface_wrapper ${TARGET_DEPENDS})
# target_link_libraries(moveit_interface_wrapper ${TARGET_DEPENDS})

install(TARGETS moveit_interface_wrapper
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()