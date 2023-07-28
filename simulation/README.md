# robotic-powder-weighting

## Getting started

### To build environments
- [install pytorch](https://pytorch.org/get-started/locally/)
- [install isaacgym](https://developer.nvidia.com/isaac-gym)
- install ROS with minimum packages (for simulation)
```
pip install --extra-index-url https://rospypi.github.io/simple rospy-all
pip install --extra-index-url https://rospypi.github.io/simple rosmaster defusedxml
```
- [install ROS (for real-robot)](http://wiki.ros.org/ROS/Installation)

## Experiments

### test powder weighing environment
This code shows powder shaking actions.
You can control the environment by joystick controller by function(test1).

```
cd simulation
python test_env.py --render
```

### learning policies
This code shows how to run the learning policies.
Particularly, isaacgym have a memory-reek error.
So, in this experiments the environment is reset by ros.
```
cd simulation
roscore or rosmaster
python root_sac.py --env Weighing --ros --ros_id 0
python Listener.py --weighing 1111111111 --ros_id 0
```
ros_id means identifier of ros nodes.
If you want to run the learning code in multiple, you have to set different ids to each experiment.
weighing means what domain parameters are fixed on learning and action definitions.
1 means random or set, 0 means fixed or removed.
All parameters are bellow.
- incline_flag
- shake_flag
- ball_radius_flag
- ball_mass_flag
- ball_friction_flag
- ball_layer_num_flag
- spoon_friction_flag
- goal_powder_amount_flag
- shake_speed_weight_flag
- gravity_flag


### make dataset
This code check the task achievement of learned policies.
Firstly, this code is run after learning.
```
cd simulation
python check_goal.py --env Weighing --test --dir policy_directly --render
```
Example: policy_directly = ./logs/Weighing/SACwithSAC/M09D29H15M14S18At_ZZ0/model/None/

### test policies on real robot
1. build up ros packages in catkin_ws
2. set up weighing machine
   - connect data cable from PC to the machine.
   - check USB port name(ex: /dev/ttyUSB or COM2)
   - set Format to 5 of weighing machine
   - set unit to mg of weighing machine
   - sudo chmod a+rw /dev/ttyUSB0
3. run the robot with learned policies
```
cd real_robot/powder_task
python PowderEnvironmentController.py --test --dir ./policy_directly/Policy.pth
```
Example: policy_directly = ./logs/Weighing/SACwithSAC/M09D29H15M14S18At_ZZ0/model/None/
