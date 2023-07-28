import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class PolicyNetworkBasic(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM, userDefinedSettings):
        super().__init__()
        self.userDefinedSettings = userDefinedSettings
        self.init_w = 3e-3
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.ACTION_DIM = ACTION_DIM

        self.linear1 = nn.Linear(STATE_DIM, userDefinedSettings.HIDDEN_NUM)
        self.linear2 = nn.Linear(userDefinedSettings.HIDDEN_NUM, userDefinedSettings.HIDDEN_NUM)
        self.linear3 = nn.Linear(userDefinedSettings.HIDDEN_NUM, userDefinedSettings.HIDDEN_NUM)

        self.mean_linear = nn.Linear(userDefinedSettings.HIDDEN_NUM, ACTION_DIM)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)

        self.log_std_linear = nn.Linear(userDefinedSettings.HIDDEN_NUM, ACTION_DIM)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)

        self.to(userDefinedSettings.DEVICE)

    def init_network(self):
        self.linear1.weight.data.uniform_(-self.init_w, self.init_w)
        self.linear1.bias.data.uniform_(-self.init_w, self.init_w)
        self.linear2.weight.data.uniform_(-self.init_w, self.init_w)
        self.linear2.bias.data.uniform_(-self.init_w, self.init_w)
        self.linear3.weight.data.uniform_(-self.init_w, self.init_w)
        self.linear3.bias.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def calc_policy(self, state):
        epsilon = 1e-6
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        deterministic_action = torch.tanh(mean)
        action = torch.tanh(mean + std * z.to(self.userDefinedSettings.DEVICE))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.userDefinedSettings.DEVICE)) - torch.log(1. - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, std, deterministic_action

    def sample_action(self, format='numpy'):
        action = np.random.uniform(low=-1., high=1., size=self.ACTION_DIM)
        return action

    def format_numpy2torch(self, data):
        return torch.FloatTensor(data).to(self.userDefinedSettings.DEVICE)

    def format_torch2numpy(self, data):
        return data.detach().cpu().numpy()
