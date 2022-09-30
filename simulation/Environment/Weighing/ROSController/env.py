import ipdb
import rospy
import std_msgs.msg
import sys
import time
import gc
import os
import numpy as np
sys.path.append('../../../')  # noqa
from Environment.Weighing.WeighingEnvironment import WeighingEnvironment  # noqa
from UserDefinedSettings import UserDefinedSettings  # noqa

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--weighing", help="parameters of weighing", type=list, default=None)
parser.add_argument("--ros_id", help="ros", type=str, default='0')
args = parser.parse_args()
weighing_parameters = np.array(args.weighing, dtype=int)


class Env:
    def __init__(self):
        self.ros_id = args.ros_id
        rospy.init_node("env" + self.ros_id)
        self.init_port2env()
        self.init_delete()
        self.init_step()
        self.init_reset()
        userDefinedSettings = UserDefinedSettings()
        self.weighingEnvironment = WeighingEnvironment(userDefinedSettings=userDefinedSettings, parameters=weighing_parameters)
        self.STATE_DIM = self.weighingEnvironment.STATE_DIM
        self.ACTION_DIM = self.weighingEnvironment.ACTION_DIM
        self.DOMAIN_PARAMETER_DIM = self.weighingEnvironment.DOMAIN_PARAMETER_DIM
        self.delete_flag = False

    def init_port2env(self):
        rospy.Subscriber("port2env_go" + self.ros_id, std_msgs.msg.String, self.callback_port2env)
        self.pub_port2env = rospy.Publisher("port2env_buck" + self.ros_id, std_msgs.msg.String, queue_size=1)

    def callback_port2env(self, msg):
        self.pub_port2env.publish("")

    def init_delete(self):
        rospy.Subscriber("delete_go" + self.ros_id, std_msgs.msg.String, self.callback_delete)
        self.pub_delete = rospy.Publisher("delete_buck" + self.ros_id, std_msgs.msg.String, queue_size=1)

    def callback_delete(self, msg):
        self.pub_delete.publish("")
        self.delete_flag = True

    def init_step(self):
        rospy.Subscriber("step_go" + self.ros_id, std_msgs.msg.Float32MultiArray, self.callback_step)
        self.pub_step = rospy.Publisher("step_buck" + self.ros_id, std_msgs.msg.Float32MultiArray, queue_size=1)

    def callback_step(self, msg):
        action = np.array(msg.data)
        next_state, reward, done, domain_parameter, task_achievement = self.weighingEnvironment.step(action, get_task_achievement=True)
        samples = np.zeros(self.STATE_DIM + self.DOMAIN_PARAMETER_DIM + 3)
        samples[:self.STATE_DIM] = next_state[:]
        samples[self.STATE_DIM] = reward
        samples[self.STATE_DIM + 1] = done
        samples[self.STATE_DIM + 2:self.STATE_DIM + 2 + self.DOMAIN_PARAMETER_DIM] = domain_parameter[:]
        samples[self.STATE_DIM + 2 + self.DOMAIN_PARAMETER_DIM] = task_achievement

        array_forPublish = std_msgs.msg.Float32MultiArray(data=samples)
        self.pub_step.publish(array_forPublish)

    def init_reset(self):
        rospy.Subscriber("reset_go" + self.ros_id, std_msgs.msg.Float32MultiArray, self.callback_reset)
        self.pub_reset = rospy.Publisher("reset_buck" + self.ros_id, std_msgs.msg.Float32MultiArray, queue_size=1)

    def callback_reset(self, msg):
        reset_info = np.array(msg.data)
        goal = reset_info[0]
        if goal > 0.:
            state = self.weighingEnvironment.reset(reset_info=[reset_info[0], 'goal_condition'])
        else:
            state = self.weighingEnvironment.reset()
        print('reset !!')
        state = std_msgs.msg.Float32MultiArray(data=state)
        self.pub_reset.publish(state)

    def run(self):
        # rospy.spin()
        while not self.delete_flag:
            time.sleep(0.001)


if __name__ == '__main__':
    env = Env()
    env.run()
