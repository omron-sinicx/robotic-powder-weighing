import ipdb
import rospy
import std_msgs.msg
import sys
import time
import gc
import subprocess
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--weighing", help="parameters of weighing", type=list, default=None)
parser.add_argument("--ros_id", help="ros", type=str, default='0')
args = parser.parse_args()


class Listener:
    """
    この実装だとuserDefinedSettingsの内容はルートからは反映されていないので動的に変更されたパラメータを変更する場合は修正が必要
    """

    def __init__(self):
        self.ros_id = args.ros_id
        rospy.init_node("listener" + self.ros_id)
        self.init_port2maker()
        self.init_make()

    def init_port2maker(self):
        rospy.Subscriber("port2maker_go" + self.ros_id, std_msgs.msg.String, self.callback_port2maker)
        self.pub_port2maker = rospy.Publisher("port2maker_buck" + self.ros_id, std_msgs.msg.String, queue_size=1)

    def init_make(self):
        rospy.Subscriber("make_go" + self.ros_id, std_msgs.msg.String, self.callback_make)
        self.pub_make_buck = rospy.Publisher("make_buck" + self.ros_id, std_msgs.msg.String, queue_size=1)

    def callback_port2maker(self, msg):
        self.pub_port2maker.publish("")

    def callback_make(self, msg):
        self.pub_make_buck.publish("")
        self.make_env()

    def make_env(self):
        weighing = ''
        for param in args.weighing:
            weighing += param
        command = ["python3", "env.py", "--weighing", weighing, "--ros_id", args.ros_id]
        # command = ["python3", "env.py"]
        subprocess.run(command)
        self.env_exist = True

    def run(self):
        rospy.spin()


if __name__ == '__main__':

    listener = Listener()
    listener.run()
