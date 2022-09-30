import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from collections import deque
# import numpy as np

from .PolicyNetworkBasic import PolicyNetworkBasic


class ActorBasic(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM, userDefinedSettings):
        super().__init__()

        self.userDefinedSettings = userDefinedSettings
        self.ACTION_DIM = ACTION_DIM
        self.policyNetwork = PolicyNetworkBasic(STATE_DIM, ACTION_DIM, userDefinedSettings)
        self.policy_optimizer = optim.Adam(self.policyNetwork.parameters(), lr=userDefinedSettings.lr)
        self.distillation_loss = nn.MSELoss()

        # if self.userDefinedSettings.LBM_flag:
        #     self.Q_list = deque(maxlen=100)
        #     self.Q_list.append(0.)

    def initialize_policy(self, target=None):
        if target is not None:
            for target_param, param in zip(self.policyNetwork.parameters(), target.policyNetwork.parameters()):
                target_param.data.copy_(param.data)
        else:
            self.policyNetwork.init_network()

    def evaluate(self, state):
        stochastic_action, log_prob, std, deterministic_action = self.policyNetwork.calc_policy(state)
        return stochastic_action, log_prob, std

    def get_action(self, state, step=None, deterministic=True, random_action_flag=False, agent_id=None):
        state = self.policyNetwork.format_numpy2torch(state)

        stochastic_action, log_prob, std, deterministic_action = self.policyNetwork.calc_policy(state)
        if deterministic:
            action = deterministic_action
        else:
            action = stochastic_action

        if random_action_flag:
            execute_action = self.policyNetwork.sample_action(format='numpy')
        else:
            execute_action = self.policyNetwork.format_torch2numpy(action)

        method_depend_info = None

        return execute_action, method_depend_info

    def update(self, alpha, log_prob, q_value):
        if self.userDefinedSettings.LBM_flag:
            # cross_entropy = torch.exp(q_value / self.userDefinedSettings.softmax_scale) * log_prob
            cross_entropy = F.softmax(q_value / self.userDefinedSettings.softmax_scale, dim=1) * log_prob
            policy_loss = (alpha * log_prob - q_value - self.userDefinedSettings.cross_entropy_weight * cross_entropy).mean()
        else:
            policy_loss = (alpha * log_prob - q_value).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss

    def direct_update(self, batch):
        """
        蒸留用関数
        調整必要
        """
        state, action, reward, next_state, done, lstm_term, domain_parameter = batch
        mean, log_std, hidden_out = self.policyNetwork.forward(state, lstm_term['last_action'], lstm_term['hidden_in'])
        student_action = torch.tanh(mean)
        teacher_action = action
        loss = self.distillation_loss(student_action, teacher_action)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss
