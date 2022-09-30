#!/usr/bin/env python3

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3

from scipy.spatial.transform import Rotation

# import isaacgym
# import torch
# import tensorflow as tf


class TouchInputClient:
    def __init__(self, calibration_mode=False):
        rospy.init_node('touch_input_client')

        self.urPosPublisher = rospy.Publisher('/end_effector_pose', PoseStamped, queue_size=5)

        rospy.Subscriber('/touch_state/joint_rotation', Vector3, self.callbackJointRotation)
        rospy.Subscriber('/touch_state/proxy_rotation', Quaternion, self.callbackProxyRotation)
        rospy.Subscriber('/touch_state/proxy_position', Point, self.callbackProxyPosition)

        self.jointRotation = Vector3()
        self.proxyRotation = Quaternion()
        self.proxyRotation.w = 1
        self.proxyPosition = Point()

        self.coordLimitUp_Touch = -10
        self.coordLimitDown_Touch = -70
        self.coordLimitForward_Touch = -50
        self.coordLimitBackward_Touch = 50

        self.coordLimitUp_Gazebo = 0.35
        self.coordLimitDown_Gazebo = 0.1
        self.coordLimitForward_Gazebo = 0.63
        self.coordLimitBackward_Gazebo = 0.268

        self.scalingUpDown = (self.coordLimitUp_Gazebo - self.coordLimitDown_Gazebo) /\
                             (self.coordLimitUp_Touch - self.coordLimitDown_Touch)
        self.scalingForwardBackward = (self.coordLimitForward_Gazebo - self.coordLimitBackward_Gazebo) /\
                                      (self.coordLimitForward_Touch - self.coordLimitBackward_Touch)

        self.calibrationMode = calibration_mode

        if self.calibrationMode:
            self.maxUpDown = -9999
            self.minUpDown = 9999
            self.maxFB = -9999
            self.minFB = 9999

        self.keyPress = False

    def callbackJointRotation(self, data):
        self.jointRotation = data

    def callbackProxyRotation(self, data):
        self.proxyRotation = data

    def callbackProxyPosition(self, data):
        self.proxyPosition = data

    def getTouchInput(self):
        jointRotation = [self.jointRotation.x, self.jointRotation.y, self.jointRotation.z]
        proxyRotationQuat = [self.proxyRotation.x, self.proxyRotation.y, self.proxyRotation.z, self.proxyRotation.w]
        proxyRotation = Rotation.from_quat(proxyRotationQuat)
        proxyRotation = proxyRotation.as_euler('xyz', degrees=True)
        proxyPosition = [self.proxyPosition.x, self.proxyPosition.y, self.proxyPosition.z]

        if self.calibrationMode:
            print()
            if proxyPosition[1] > self.maxUpDown:
                self.maxUpDown = proxyPosition[1]
            elif proxyPosition[1] < self.minUpDown:
                self.minUpDown = proxyPosition[1]

            if proxyPosition[2] > self.maxFB:
                self.maxFB = proxyPosition[2]
            elif proxyPosition[2] < self.minFB:
                self.minFB = proxyPosition[2]

            print("Up/Down:", proxyPosition[1], '; min:', self.minUpDown, '; max:', self.maxUpDown)
            print("Forward/Backward:", proxyPosition[2], '; min:', self.minFB, '; max:', self.maxFB)
        return [proxyPosition[1], proxyPosition[2], proxyRotation[2]]

    def main(self):
        r = rospy.Rate(10)
        prevUpDown = 0
        prevForwardBackward = 0
        while not rospy.is_shutdown():
            touchInput = self.getTouchInput()
            _publish = True
            if touchInput != None:
                upDown, forwardBackward, orientation = touchInput
                if upDown < self.coordLimitDown_Touch:
                    upDown = self.coordLimitDown_Touch
                if upDown > self.coordLimitUp_Touch:
                    upDown = self.coordLimitUp_Touch
                if forwardBackward < self.coordLimitForward_Touch:
                    forwardBackward = self.coordLimitForward_Touch
                if forwardBackward > self.coordLimitBackward_Touch:
                    forwardBackward = self.coordLimitBackward_Touch

                # Map Touch coordinate to Gazebo coordinate
                upDown = ((upDown - self.coordLimitDown_Touch) * self.scalingUpDown) + self.coordLimitDown_Gazebo
                forwardBackward = ((forwardBackward - self.coordLimitBackward_Touch) * self.scalingForwardBackward) + self.coordLimitBackward_Gazebo
                quatOrientation = Rotation.from_euler('xyz', [-(orientation - 90), 0, 90], degrees=True).as_quat()
                # print(quatOrientation)

                if upDown == prevUpDown and forwardBackward == prevForwardBackward:
                    _publish = False
                else:
                    prevUpDown = upDown
                    prevForwardBackward = forwardBackward

                pose = PoseStamped()
                pose.pose.position.x = forwardBackward
                pose.pose.position.y = 0.2
                pose.pose.position.z = upDown
                pose.pose.orientation.x = quatOrientation[0]
                pose.pose.orientation.y = quatOrientation[1]
                pose.pose.orientation.z = quatOrientation[2]
                pose.pose.orientation.w = quatOrientation[3]

                print('\nUp / Down          :', upDown)
                print('Forward / Backward :', forwardBackward)

                if _publish:
                    self.urPosPublisher.publish(pose)
            r.sleep()


if __name__ == "__main__":
    touchInputClient = TouchInputClient(calibration_mode=False)
    try:
        touchInputClient.main()
    except rospy.ROSInterruptException:
        pass
