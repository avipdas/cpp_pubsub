cmake_minimum_required(VERSION 3.5)
project(cpp_pubsub)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
enable_language(CUDA)


# Original talker and listener
add_executable(talker publisher_lambda_function.cpp)
target_link_libraries(talker PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

add_executable(listener subscriber_lambda_function.cpp)
target_link_libraries(listener PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

# New GPS publisher
add_executable(gps_publisher src/gps_publisher.cpp)
ament_target_dependencies(gps_publisher rclcpp sensor_msgs)

# New IMU publisher
add_executable(imu_publisher src/imu_publisher.cpp)
ament_target_dependencies(imu_publisher rclcpp sensor_msgs)

# New multi-subscriber
add_executable(multi_subscriber src/multi_subscriber.cpp)
ament_target_dependencies(multi_subscriber rclcpp sensor_msgs)

add_executable(fusion_node src/fusion_node.cpp)
ament_target_dependencies(fusion_node rclcpp sensor_msgs)

# Compile CUDA kernel
add_library(ekf_update STATIC cuda/ekf_update.cu)
set_target_properties(ekf_update PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ekf_node src/ekf_node.cpp)
find_package(Eigen3 REQUIRED)
ament_target_dependencies(ekf_node rclcpp sensor_msgs geometry_msgs)
target_link_libraries(ekf_node Eigen3::Eigen ekf_update)

# Install all executables
install(TARGETS
  talker
  listener
  gps_publisher
  imu_publisher
  multi_subscriber
  fusion_node
  ekf_node

  DESTINATION lib/${PROJECT_NAME})

ament_package()
