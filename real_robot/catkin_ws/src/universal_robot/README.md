# Universal Robot

This package is for UR3 and UR5, not for e-series.

## Installation

```
cd $HOME/catkin_ws/src
git clone https://github.com/naoteen/Universal_Robots_ROS_Driver
git clone -b A111 https://github.com/naoteen/universal_robot.git
cd ..

# building
catkin_make

# activate this workspace
source $HOME/catkin_ws/devel/setup.bash
```


## Robot environment

### Real
Don't forget to source the correct setup shell files and use a new terminal for each command!   

To bring up the real robot, run:

```roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=IP_OF_THE_ROBOT [reverse_port:=REVERSE_PORT]```

CAUTION:  
Remember that you should always have your hands on the big red button in case there is something in the way or anything unexpected happens.


### Simulation
To bring up the simulated robot in Gazebo, run:

```roslaunch ur_gazebo ur5.launch```


## Moveit and rviz
You can use MoveIt! to control the robot.  
There exist MoveIt! configuration packages for both robots.  

For setting up the MoveIt! nodes to allow motion planning run:

```roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true```

For starting up RViz with a configuration including the MoveIt! Motion Planning plugin run:

```roslaunch ur5_moveit_config moveit_rviz.launch config:=true```

NOTE:  
If you use real robot, ``sim:=false``



## Joint limit 

### Unchained ver. (not recommended)
Default setting is a joint_limited version using joint limits restricted to [0,pi]. In order to use this full joint limits [-2pi, 2pi], simply use the launch file arguments 'limited', i.e.:  

```roslaunch ur_gazebo ur5.launch limited:=false```

```roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true limited:=false```

```roslaunch ur5_moveit_config moveit_rviz.launch config:=true```


### How to set angle for joint limit
Editing  joint limits at ```universal_robot/ur_description/urdf/ur5_joint_limited_robot.urdf.xacro```.

```
  <!-- arm -->
  <xacro:ur5_robot prefix="" joint_limited="true"
    shoulder_pan_lower_limit="${-pi/4.0}" shoulder_pan_upper_limit="${pi/4.0}"
    shoulder_lift_lower_limit="${-pi}" shoulder_lift_upper_limit="${pi*0.0}"
    elbow_joint_lower_limit="${-pi*0.0}" elbow_joint_upper_limit="${pi}"
    wrist_1_lower_limit="${-pi*1.3}" wrist_1_upper_limit="${-pi/3.0}"
    wrist_2_lower_limit="${-pi}" wrist_2_upper_limit="${pi*0.0}"
    wrist_3_lower_limit="${-pi}" wrist_3_upper_limit="${pi}"
    transmission_hw_interface="$(arg transmission_hw_interface)"
  />
```
