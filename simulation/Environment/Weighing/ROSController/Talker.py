import rospy
import std_msgs.msg
import time
import numpy as np
import sys

sys.path.append('../../../')  # noqa
from Environment.Weighing.WeighingEnvironment import WeighingEnvironment  # noqa
# from UserDefinedSettings import UserDefinedSettings  # noqa


class Talker:
    def __init__(self, userDefinedSettings):
        self.weighingEnvironment = WeighingEnvironment(userDefinedSettings=userDefinedSettings)
        self.STATE_DIM = self.weighingEnvironment.STATE_DIM
        self.ACTION_DIM = self.weighingEnvironment.ACTION_DIM
        self.DOMAIN_PARAMETER_DIM = self.weighingEnvironment.DOMAIN_PARAMETER_DIM
        self.ros_id = userDefinedSettings.ros_id

        rospy.init_node("talker" + self.ros_id)
        self.init_port2maker()
        self.init_make()
        self.init_port2env()
        self.init_delete()
        self.init_step()
        self.init_reset()

        self.port2maker()

    def init_port2maker(self):
        rospy.Subscriber("port2maker_buck" + self.ros_id, std_msgs.msg.String, self.callback_port2maker)
        self.pub_port2maker = rospy.Publisher("port2maker_go" + self.ros_id, std_msgs.msg.String, queue_size=1)
        self.port2maker_ready = False

    def callback_port2maker(self, msg):
        self.port2maker_ready = True

    def port2maker(self):
        while not rospy.is_shutdown() and not self.port2maker_ready:
            self.pub_port2maker.publish("")
            time.sleep(0.01)
        print('port2maker connected !!')

    def init_make(self):
        rospy.Subscriber("make_buck" + self.ros_id, std_msgs.msg.String, self.callback_make)
        self.pub = rospy.Publisher("make_go" + self.ros_id, std_msgs.msg.String, queue_size=1)
        self.make_done = False

    def callback_make(self, msg):
        self.make_done = True

    def make(self):
        while not rospy.is_shutdown() and not self.make_done:
            self.pub.publish("")
            time.sleep(0.01)
        time.sleep(2)  # make するまで目測タイマーで待っている=要修正
        print('make done !!')
        self.make_done = False
        self.delete_ready = False

    def init_port2env(self):
        rospy.Subscriber("port2env_buck" + self.ros_id, std_msgs.msg.String, self.callback_port2env)
        self.pub_port2env = rospy.Publisher("port2env_go" + self.ros_id, std_msgs.msg.String, queue_size=1)
        self.port2env_ready = False

    def callback_port2env(self, msg):
        self.port2env_ready = True

    def port2env(self):
        while not rospy.is_shutdown() and not self.port2env_ready:
            self.pub_port2env.publish("")
            time.sleep(0.01)
        time.sleep(1.5)
        print('port2env connected !!')

    def init_delete(self):
        rospy.Subscriber("delete_buck" + self.ros_id, std_msgs.msg.String, self.callback_delete)
        self.pub_delete = rospy.Publisher("delete_go" + self.ros_id, std_msgs.msg.String, queue_size=1)
        self.delete_ready = False

    def callback_delete(self, msg):
        self.delete_ready = True

    def delete(self):
        while not rospy.is_shutdown() and not self.delete_ready:
            self.pub_delete.publish("")
            time.sleep(0.01)
        time.sleep(1)
        print('deleted !!')

    def init_step(self):
        rospy.Subscriber("step_buck" + self.ros_id, std_msgs.msg.Float32MultiArray, self.callback_step)
        self.pub_step = rospy.Publisher("step_go" + self.ros_id, std_msgs.msg.Float32MultiArray, queue_size=1)

    def callback_step(self, msg):
        samples = np.array(msg.data)
        self.next_state = samples[:self.STATE_DIM]
        self.reward = samples[self.STATE_DIM]
        self.done = samples[self.STATE_DIM + 1]
        self.domain_parameter = samples[self.STATE_DIM + 2:self.STATE_DIM + 2 + self.DOMAIN_PARAMETER_DIM]
        self.task_achievement = samples[self.STATE_DIM + 2 + self.DOMAIN_PARAMETER_DIM]

    def step(self, action):
        array_forPublish = std_msgs.msg.Float32MultiArray(data=action)
        self.pub_step.publish(array_forPublish)
        rospy.wait_for_message('step_buck' + self.ros_id, std_msgs.msg.Float32MultiArray, timeout=None)
        print('step !!')
        return self.next_state, self.reward, self.done, self.domain_parameter, self.task_achievement

    def init_reset(self):
        rospy.Subscriber("reset_buck" + self.ros_id, std_msgs.msg.Float32MultiArray, self.callback_reset)
        self.pub_reset = rospy.Publisher("reset_go" + self.ros_id, std_msgs.msg.Float32MultiArray, queue_size=1)

    def callback_reset(self, msg):
        samples = np.array(msg.data)
        self.next_state = samples[0:self.STATE_DIM]

    def reset(self, reset_info):
        array_forPublish = std_msgs.msg.Float32MultiArray(data=reset_info)
        self.pub_reset.publish(array_forPublish)
        rospy.wait_for_message('reset_buck' + self.ros_id, std_msgs.msg.Float32MultiArray, timeout=None)
        print('reset done!!')
        return self.next_state


if __name__ == '__main__':
    talker = Talker()

    for i in range(10):
        print(i)
        talker.make()
        talker.port2env()
        state = talker.reset()
        print('state', state)
        for j in range(3):
            action = np.array([0, 0])
            next_state, reward, done, domain_parameter, task_achievement = talker.step(action)
            print('reward', reward)
        talker.delete()

"""
タスク：
1)ドメインパラメータのランダマイズが関数を再実行するたびにシードが同じなので同じパラメータが出てくる
現状では乱数を固定しないことで解決している
2)現状だと待ち時間が長いので固定待ち時間を消してwait_for_messageにする
3)現状のDRフレームワークと環境を統合させて学習を回す
4)学習が回るようなら最終チェックでおかしな点を探す
学習フレームワークで動かすとdelでエラーがでるが現状動作に影響ないので後回し
"""
