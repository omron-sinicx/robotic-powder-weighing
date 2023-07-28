#!/usr/bin/env python3

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation
import socket
import select

import isaacgym
import torch
import tensorflow as tf

def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + '\n%s\n', data.actual.positions)
    rospy.loginfo(data.actual.positions)

def positionListener():
    rospy.init_node('ur_listener', anonymous=True)
    rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, callback)
    rospy.spin()

def positionSetter():
    rospy.init_node('ur_setter', anonymous=True)
    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

    traj = JointTrajectory()
    traj.header = Header()
    traj.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                        'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                        'wrist_3_joint']

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        traj.header.stamp = rospy.Time.now()
        pts = JointTrajectoryPoint()
        pts.positions = [ 0.0,
                         -0.785398163,
                          1.570796327,
                         -2.35619449,
                          0.0,
                          0.0]
        pts.time_from_start = rospy.Duration(0.5)
        traj.points = [pts]

        pub.publish(traj)
        rate.sleep()

def getTouchInput():
    # positionSetter()
    ready = select.select([server_socket], [], [], 1e-10)
    if ready[0]:
        # data = server_socket.recv(4096)
        data = server_socket.recv(1024).decode('UTF-8').split(';')

        parsed_data=[]
        for i in data:
            temp=[]
            for j in i.split(' '):
                temp.append(float(j))
            parsed_data.append(temp)
        
        joint_rotation = parsed_data[0]
        proxy_rotation = Rotation.from_quat(parsed_data[1])
        proxy_rotation = proxy_rotation.as_euler('xyz', degrees=True)
        proxy_position = parsed_data[2]

        print('\nJoint Rotation', joint_rotation,'\nProxy Rotation',  proxy_rotation,'\nProxy Position',  proxy_position)

if __name__ == '__main__':
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('', 8080))
        server_socket.setblocking(0)

        while(1):
            getTouchInput()
        
        # print("test")
    except rospy.ROSInterruptException:
        pass