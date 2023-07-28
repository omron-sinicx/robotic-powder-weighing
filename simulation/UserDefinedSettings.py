import os
import argparse
import socket
from isaacgym import gymutil  # noqa
from isaacgym import gymapi  # noqa
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="learning flag", action="store_true")
parser.add_argument("--seed", help="seed", type=int, default=1)
parser.add_argument("--dir", help="directory of tested policy", type=str, default=None)
parser.add_argument("--num", help="model_num_in_distillation", type=str, default=0)
parser.add_argument("--alpha", help="mixture rate of distillation", type=float, default=-1.)
parser.add_argument("--gpu", help="gpu num", type=str, default='0')
parser.add_argument("--env", help="env name", type=str, default='Weighing')
parser.add_argument("--render", help="render", action="store_true")
parser.add_argument("--save_image", help="save image", action="store_true")
parser.add_argument("--network", help="network", type=str, default='basic')
parser.add_argument("--path", help="header of save directory", type=str, default=None)
parser.add_argument("--flag", help="parameter flag (10)", type=list, default='1111111111')
parser.add_argument("--episode", help="current episode num", type=int)
parser.add_argument("--goal", help="goal", type=float)


args = parser.parse_args()


class UserDefinedSettings(object):

    def __init__(self, BASE_RL_METHOD='SAC'):

        self.DEVICE = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        self.ENVIRONMENT_NAME = args.env
        dir_name = 'test'
        self.HOST_NAME = socket.gethostname()
        self.LOG_DIRECTORY = os.path.join(os.environ['HOME'], self.HOST_NAME, 'logs', self.ENVIRONMENT_NAME, BASE_RL_METHOD, dir_name)

        self.LSTM_FLAG = True
        self.DOMAIN_RANDOMIZATION_FLAG = True

        self.BASE_RL_METHOD = BASE_RL_METHOD
        self.seed = args.seed
        self.save_image = args.save_image

        self.num_steps = 1e6
        self.batch_size = 16
        self.policy_update_start_episode_num = 20
        self.learning_episode_num = 20
        self.total_episode_num = 4000

        self.learning_rate = self.lr = 1e-4

        self.HIDDEN_NUM = 128
        self.onPolicy_distillation = True
        self.entropy_tuning_scale = 1.

        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.updates_per_step = 1
        self.target_update_interval = 1  # episode num
        self.evaluate_interval = 10  # episode num
        self.initializer = 'xavier'
        self.run_num_per_evaluate = 3
        self.average_num_for_model_save = self.run_num_per_evaluate
        self.LEARNING_REWARD_SCALE = 1.
        self.MODEL_SAVE_INDEX = 'test'  # test, train

        self.ACTION_DISCRETE_FLAG = False

        self.TEST_FLAG = args.test
        if self.TEST_FLAG:
            self.TEST_DIR = args.dir

        self.RENDER_FLAG = args.render
        self.network_type = args.network

        self.current_episode_num = args.episode

        self.TEST_DIR = args.dir

        self.goal = args.goal

        self.flag = args.flag
