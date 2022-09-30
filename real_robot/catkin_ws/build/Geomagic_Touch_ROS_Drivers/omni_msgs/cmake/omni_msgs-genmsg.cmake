# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "omni_msgs: 3 messages, 0 services")

set(MSG_I_FLAGS "-Iomni_msgs:/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(omni_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_custom_target(_omni_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "omni_msgs" "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" "geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_custom_target(_omni_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "omni_msgs" "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" ""
)

get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_custom_target(_omni_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "omni_msgs" "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" "geometry_msgs/Quaternion:geometry_msgs/Point:geometry_msgs/Vector3:geometry_msgs/Pose:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs
)
_generate_msg_cpp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs
)
_generate_msg_cpp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(omni_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(omni_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(omni_msgs_generate_messages omni_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_cpp _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_cpp _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_cpp _omni_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(omni_msgs_gencpp)
add_dependencies(omni_msgs_gencpp omni_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS omni_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs
)
_generate_msg_eus(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs
)
_generate_msg_eus(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(omni_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(omni_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(omni_msgs_generate_messages omni_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_eus _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_eus _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_eus _omni_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(omni_msgs_geneus)
add_dependencies(omni_msgs_geneus omni_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS omni_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs
)
_generate_msg_lisp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs
)
_generate_msg_lisp(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(omni_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(omni_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(omni_msgs_generate_messages omni_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_lisp _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_lisp _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_lisp _omni_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(omni_msgs_genlisp)
add_dependencies(omni_msgs_genlisp omni_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS omni_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs
)
_generate_msg_nodejs(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs
)
_generate_msg_nodejs(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(omni_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(omni_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(omni_msgs_generate_messages omni_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_nodejs _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_nodejs _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_nodejs _omni_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(omni_msgs_gennodejs)
add_dependencies(omni_msgs_gennodejs omni_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS omni_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs
)
_generate_msg_py(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs
)
_generate_msg_py(omni_msgs
  "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(omni_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(omni_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(omni_msgs_generate_messages omni_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniFeedback.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_py _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniButtonEvent.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_py _omni_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yuki-kadokawa/catkin_ws/src/Geomagic_Touch_ROS_Drivers/omni_msgs/msg/OmniState.msg" NAME_WE)
add_dependencies(omni_msgs_generate_messages_py _omni_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(omni_msgs_genpy)
add_dependencies(omni_msgs_genpy omni_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS omni_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/omni_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(omni_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(omni_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(omni_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/omni_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(omni_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(omni_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(omni_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/omni_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(omni_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(omni_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(omni_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/omni_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(omni_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(omni_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(omni_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/omni_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(omni_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(omni_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(omni_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
