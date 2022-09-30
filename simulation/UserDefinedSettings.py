import os
import warnings
import datetime

import numpy as np
import random
import argparse
import socket

from isaacgym import gymutil
from isaacgym import gymapi
import torch

warnings.simplefilter('ignore', FutureWarning)  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="learning flag", action="store_true")
parser.add_argument("--seed", help="seed", type=int, default=1)
parser.add_argument("--dir", help="directory of tested policy", type=str)
parser.add_argument("--gpu", help="gpu num", type=str, default='0')
parser.add_argument("--env", help="env name", type=str, default='HalfCheetah')
parser.add_argument("--render", help="render", action="store_true")
parser.add_argument("--save_image", help="save image", action="store_true")
parser.add_argument("--path", help="header of save directory", type=str, default=None)
parser.add_argument("--notDR", help="DR", action="store_true")
parser.add_argument("--ros", help="ros", action="store_true")
parser.add_argument("--ros_id", help="ros", type=str, default='0')
parser.add_argument("--weighing", help="weighing", type=list)

args = parser.parse_args()


class UserDefinedSettings(object):

    def __init__(self, LEARNING_METHOD='_', BASE_RL_METHOD='SAC'):

        self.DEVICE = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        self.ros_id = args.ros_id
        self.ENVIRONMENT_NAME = args.env
        current_time = datetime.datetime.now()
        file_name = 'M{:0=2}D{:0=2}H{:0=2}M{:0=2}S{:0=2}At_{}{}'.format(current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second, socket.gethostname(), self.ros_id)
        self.HOST_NAME = socket.gethostname()
        if args.path == 'local':
            self.LOG_DIRECTORY = os.path.join(os.environ['HOME'], self.HOST_NAME, 'logs', self.ENVIRONMENT_NAME, LEARNING_METHOD + 'with' + BASE_RL_METHOD, file_name)
        else:
            self.LOG_DIRECTORY = os.path.join('logs', self.ENVIRONMENT_NAME, LEARNING_METHOD + 'with' + BASE_RL_METHOD, file_name)
        self.LSTM_FLAG = True
        self.DOMAIN_RANDOMIZATION_FLAG = not args.notDR
        self.BASE_RL_METHOD = BASE_RL_METHOD
        self.seed = args.seed
        self.ros = args.ros
        self.save_image = args.save_image

        self.num_steps = 1e6
        self.batch_size = 16
        self.policy_update_start_episode_num = 15
        self.learning_episode_num = 5101
        self.lr = 1e-4
        self.learning_rate = self.lr
        self.HIDDEN_NUM = 128
        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_tuning_scale = 1  # 1
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.updates_per_step = 1
        self.target_update_interval = 1  # episode num
        self.evaluate_interval = 10  # episode num
        self.initializer = 'xavier'
        self.run_num_per_evaluate = 3  # 5
        self.average_num_for_model_save = self.run_num_per_evaluate
        self.LEARNING_REWARD_SCALE = 1.
        self.MODEL_SAVE_INDEX = 'test'  # test, train

        self.ACTION_DISCRETE_FLAG = False

        self.TEST_FLAG = args.test
        if self.TEST_FLAG:
            self.TEST_DIR = args.dir

        self.RENDER_FLAG = args.render
