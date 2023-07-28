import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class PolicyNetworkLSTM(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM, userDefinedSettings):
        super().__init__()
        self.userDefinedSettings = userDefinedSettings
        self.init_w = 3e-3
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.ACTION_DIM = ACTION_DIM

        self.linear1 = nn.Linear(STATE_DIM, userDefinedSettings.HIDDEN_NUM)
        self.linear2 = nn.Linear(STATE_DIM + ACTION_DIM, userDefinedSettings.HIDDEN_NUM)
        self.lstm1 = nn.LSTM(userDefinedSettings.HIDDEN_NUM, userDefinedSettings.HIDDEN_NUM, batch_first=True)
        self.linear3 = nn.Linear(2 * userDefinedSettings.HIDDEN_NUM, userDefinedSettings.HIDDEN_NUM)
        self.linear4 = nn.Linear(userDefinedSettings.HIDDEN_NUM, userDefinedSettings.HIDDEN_NUM)

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

        self.lstm1.weight_ih_l0.data.uniform_(-self.init_w, self.init_w)
        self.lstm1.weight_hh_l0.data.uniform_(-self.init_w, self.init_w)
        self.lstm1.bias_ih_l0.data.uniform_(-self.init_w, self.init_w)
        self.lstm1.bias_hh_l0.data.uniform_(-self.init_w, self.init_w)

        self.linear3.weight.data.uniform_(-self.init_w, self.init_w)
        self.linear3.bias.data.uniform_(-self.init_w, self.init_w)
        self.linear4.weight.data.uniform_(-self.init_w, self.init_w)
        self.linear4.bias.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state, last_action, hidden_in):
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, hidden_out = self.lstm1(lstm_branch, hidden_in)
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std, hidden_out

    def calc_policy(self, state, last_action=None, hidden_in=None):

        epsilon = 1e-6
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        deterministic_action = torch.tanh(mean)
        action = torch.tanh(mean + std * z.to(self.userDefinedSettings.DEVICE))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.userDefinedSettings.DEVICE)) - torch.log(1. - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, std, deterministic_action, hidden_out

    def sample_action(self, format='numpy'):
        action = np.random.uniform(low=-1., high=1., size=self.ACTION_DIM)
        return action

    def format_numpy2torch(self, data):
        return torch.FloatTensor(data).unsqueeze(0).unsqueeze(0).to(self.userDefinedSettings.DEVICE)

    def format_torch2numpy(self, data):
        return data.detach().cpu().numpy()[0][0]
