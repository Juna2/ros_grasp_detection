cmake_minimum_required(VERSION 2.8.3)
project(ros_grasp_detection)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs)

add_message_files(DIRECTORY msg FILES Floats.msg HeaderString.msg positionNpose.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs)

# do not wildcard install files since the root folder of the package will contain a debian folder for releasing

install(FILES
  src/headers.launch
  src/test.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)


