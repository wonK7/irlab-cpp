cmake_minimum_required(VERSION 3.8)
project(spot_schemas_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/Status.msg"
  "msg/Object2D.msg"
  "msg/ObjectPose3D.msg"
  "msg/GraspStatus.msg"
  "msg/ColorLabel.msg"
  
  # Services
  "srv/DetectObjectColor.srv"
  "srv/Rotate.srv"
  "srv/WalkToPose.srv"
  "srv/Walk.srv"
  "srv/GraspHand.srv"
  "srv/ReleaseHand.srv"

  DEPENDENCIES geometry_msgs
)

ament_package()
