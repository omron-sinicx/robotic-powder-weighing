#!/usr/bin/env python
import moveit_commander
import rospy
import copy
import math
import geometry_msgs.msg


class moveitController(object):
    def __init__(self):
        self.angle_reset = [math.pi / 4, -math.pi / 2, math.pi / 2, -math.pi, -math.pi / 2, 0]
        self.angle_init = [math.pi / 4, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0]
        self.robot = moveit_commander.RobotCommander()
        print(self.robot.get_current_state())
        self.mani = moveit_commander.MoveGroupCommander("manipulator")

    def setJointAngles(self, angle):
        self.mani.clear_pose_targets()
        self.mani.set_joint_value_target(angle)
        self.mani.go()

    def setPosition(self, position):
        pose = self.mani.get_current_pose().pose
        pose.position = position
        plan = self.generate2pointsPath(pose)
        self.mani.execute(plan)

    def setPose(self, pose):
        # self.mani.clear_pose_targets()
        # self.mani.set_pose_target(pose)
        # self.mani.go()
        plan = self.generate2pointsPath(pose)
        self.mani.execute(plan)

    def moveCartesianSpace(self, x, y, z):
        pose = self.mani.get_current_pose().pose
        pose.position.x += x / math.sqrt(2)
        pose.position.y += x / math.sqrt(2)
        pose.position.x -= y / math.sqrt(2)
        pose.position.y += y / math.sqrt(2)
        pose.position.z += z
        self.setPose(pose)

    def generate2pointsPath(self, goal):
        # self.mani.clear_pose_targets()
        # self.mani.set_pose_target(start)
        # self.mani.go()
        self.mani.set_pose_target(goal)
        plan = self.mani.plan()
        return plan

    def generate3pointsPath(self, start, wpose, goal):
        self.mani.clear_pose_targets()
        self.mani.set_pose_target(start)
        self.mani.go()
        waypoints = []
        waypoints.append(start)
        waypoints.append(wpose)
        waypoints.append(goal)
        (plan, fraction) = self.mani.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan

    def generate4pointsPath(self, start, wpose_1, wpose_2, goal):
        self.mani.clear_pose_targets()
        self.mani.set_pose_target(start)
        self.mani.go()
        waypoints = []
        waypoints.append(start)
        waypoints.append(wpose_1)
        waypoints.append(wpose_2)
        waypoints.append(goal)
        (plan, fraction) = self.mani.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan

    def resetPose(self):
        self.setJointAngles(self.angle_reset)

    def initPose(self):
        self.setJointAngles(self.angle_init)


if __name__ == '__main__':
    rospy.init_node("UR5_operator")
    UR5 = moveitController()

    # stable moveit plan
    # print "Reset"
    # UR5.resetPose()
    # UR5.initPose()

    # UR5.moveCartesianSpace(0.1, 0.4, -0.3)

    pose_ready = UR5.mani.get_current_pose().pose
    print(pose_ready)
    # pose_ready.position.z -= 0.1

    # pose_mid = copy.deepcopy(pose_ready)
    # pose_mid.position.x -= 0.1
    # pose_mid.position.y -= 0.1

    # pose_tgt = copy.deepcopy(pose_mid)
    # pose_tgt.position.x -= 0.1
    # pose_tgt.position.y -= 0.1

    # pose_wpose = copy.deepcopy(pose_tgt)
    # pose_wpose.position.z += 0.1

    # trace = UR5.generate3pointsPath(pose_ready, pose_mid, pose_tgt)
    # # trace = UR5.generate2pointsPath(pose_ready, pose_tgt)

    # print "Ready..."
    # UR5.setPose(pose_ready)
    # while not rospy.is_shutdown():
    #     print "Trace"
    #     UR5.mani.execute(trace)
    #     rospy.sleep(1)
    #     print "Back"
    #     UR5.moveCartesianSpace(0, 0, 0.05)
    #     UR5.setPose(pose_ready)
    #     rospy.sleep(1)
