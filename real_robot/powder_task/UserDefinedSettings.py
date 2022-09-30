import ipdb
import os
# import sys
import warnings
import datetime

import numpy as np
import random
import argparse
import socket

import torch

warnings.simplefilter('ignore', FutureWarning)  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="learning flag", action="store_true")
parser.add_argument("--LBM", help="LBM flag", action="store_true")
parser.add_argument("--ip", help="count of remote ip address", type=int, default=0)
parser.add_argument("--seed", help="seed", type=int, default=1)
parser.add_argument("--dir", help="directory of tested policy", type=str)
parser.add_argument("--num", help="model_num_in_distillation", type=str, default=0)
parser.add_argument("--alpha", help="mixture rate of distillation", type=float, default=-1.)
parser.add_argument("--gpu", help="gpu num", type=str, default='0')
parser.add_argument("--env", help="env name", type=str, default='HalfCheetah')
parser.add_argument("--render", help="render", action="store_true")
parser.add_argument("--network", help="network", type=str, default='basic')
parser.add_argument("--dnum", help="domain num", type=int, default=6)
parser.add_argument("--path", help="header of save directory", type=str, default=None)
args = parser.parse_args()


seed_number = args.seed
os.environ['PYTHONHASHSEED'] = str(seed_number)
np.random.seed(seed_number)
random.seed(seed_number)
torch.manual_seed(seed_number)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.deterministic = True


class UserDefinedSettings(object):

    def __init__(self, LEARNING_METHOD='_', BASE_RL_METHOD='SAC'):
        # LBM #####
        self.LBM_flag = args.LBM
        self.cross_entropy_weight = 0.01
        self.softmax_scale = 1  # 1.
        if self.LBM_flag:
            LEARNING_METHOD = "LBM" + LEARNING_METHOD
        ######

        self.DEVICE = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        self.ENVIRONMENT_NAME = args.env  # Pendulum, Weighing, Swimmer, Pusher, SwingUp, Hopper, HalfCheetah, Excavator_v1, Excavator_v2, Excavator_v3, Excavator_v4, 2DoF, LunarLander, SandOperator
        current_time = datetime.datetime.now()
        file_name = 'M{:0=2}D{:0=2}H{:0=2}M{:0=2}S{:0=2}At_{}'.format(current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second, socket.gethostname())
        self.HOST_NAME = socket.gethostname()
        if args.path == 'local':
            self.LOG_DIRECTORY = os.path.join(os.environ['HOME'], self.HOST_NAME, 'logs', self.ENVIRONMENT_NAME, LEARNING_METHOD + 'with' + BASE_RL_METHOD, file_name)
        else:
            self.LOG_DIRECTORY = os.path.join('logs', self.ENVIRONMENT_NAME, LEARNING_METHOD + 'with' + BASE_RL_METHOD, file_name)
        self.LSTM_FLAG = True
        self.DOMAIN_RANDOMIZATION_FLAG = False
        self.BASE_RL_METHOD = BASE_RL_METHOD  # SAC, CPI, DQN, (root関数で定義)
        self.seed = args.seed

        self.num_steps = 1e6
        if True:
            self.batch_size = 16  # 16
            self.policy_update_start_episode_num = 15  # 30
            self.learning_episode_num = 300  # 150
            # self.rollout_cycle_num = 160  # 240
        else:
            # for debug
            self.batch_size = 2
            self.policy_update_start_episode_num = 1
            self.learning_episode_num = 2
            self.rollout_cycle_num = 3

        self.lr = 1e-4
        self.learning_rate = self.lr

        # if self.ENVIRONMENT_NAME == 'Pendulum':
        #     self.HIDDEN_NUM = 64
        #     self.GLOBAL_DIST_ITERATION_NUM = 30  # 30
        #     self.DOMAIN_NUM = 4  # 4
        #     self.check_global_interbal = 6
        #     self.rollout_cycle_num = 1500  # 320:divide 500:DnC
        #     if self.DOMAIN_NUM < 4:
        #         self.rollout_cycle_num = int(self.rollout_cycle_num * (4 / self.DOMAIN_NUM))
        #     self.check_global_flag = True
        #     self.onPolicy_distillation = True
        if self.ENVIRONMENT_NAME == 'Pendulum':
            self.HIDDEN_NUM = 64
            self.GLOBAL_DIST_ITERATION_NUM = 30  # 30
            self.DOMAIN_NUM = 4  # 4
            self.check_global_interbal = 1
            self.rollout_cycle_num = 320  # 320:divide 500:DnC
            if self.DOMAIN_NUM < 4:
                self.rollout_cycle_num = int(self.rollout_cycle_num * (4 / self.DOMAIN_NUM))
            self.check_global_flag = True
            self.onPolicy_distillation = True
        elif self.ENVIRONMENT_NAME == 'SandOperator':
            self.HIDDEN_NUM = 128
            self.GLOBAL_DIST_ITERATION_NUM = 100
            self.DOMAIN_NUM = 6
            self.check_global_interbal = 6
            self.rollout_cycle_num = 1000  # 240
            self.check_global_flag = False  # グローバル方策を作るか確認するか
            self.onPolicy_distillation = False
        else:
            self.HIDDEN_NUM = 128
            self.GLOBAL_DIST_ITERATION_NUM = 100
            self.DOMAIN_NUM = 6
            self.check_global_interbal = 6  # グローバル方策を作成と検証する周期
            self.rollout_cycle_num = 1000  # 240
            self.check_global_flag = True  # グローバル方策を作るか確認するか
            self.onPolicy_distillation = True

        print('rollout-cycle:', self.rollout_cycle_num)
        print('domain-num:', self.DOMAIN_NUM)
        print('Env:', self.ENVIRONMENT_NAME)
        print('Hidden-num:', self.HIDDEN_NUM)
        print('global-ite:', self.GLOBAL_DIST_ITERATION_NUM)
        print('domain-num:', self.DOMAIN_NUM)
        print('check-global:', self.check_global_flag)

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
        # self.learning_episode_num_all_domain = 5000
        self.LEARNING_REWARD_SCALE = 1.
        self.MODEL_SAVE_INDEX = 'test'  # test, train

        # distillation parameters
        ############################################################
        # バッファをTrueにしたのをFalseに直す
        self.set_policy_mixture_rate = args.alpha  # 0~1: const. , -1: var, over +1: random-alpha ############################################################
        print('alpha', self.set_policy_mixture_rate)
        ############################################################
        self.value_init_flag = True
        self.policy_init_flag = False
        self.model_num_in_distillation = args.num  # for test

        self.ACTION_DISCRETE_FLAG = False

        self.REMOTE_IP = args.ip
        self.TEST_FLAG = args.test
        if self.TEST_FLAG:
            self.TEST_DIR = args.dir

        self.RENDER_FLAG = args.render
        self.network_type = args.network
