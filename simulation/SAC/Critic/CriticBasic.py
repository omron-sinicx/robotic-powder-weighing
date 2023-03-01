import torch
import torch.nn as nn
import torch.optim as optim

from .QNetworkBasic import QNetworkBasic


class CriticBasic(object):

    def __init__(self, state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings):
        self.userDefinedSettings = userDefinedSettings
        self.soft_q_net1 = QNetworkBasic(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.soft_q_net2 = QNetworkBasic(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.target_soft_q_net1 = QNetworkBasic(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.target_soft_q_net2 = QNetworkBasic(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.network_initialization()

    def network_initialization(self):
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.SmoothL1Loss()
        self.soft_q_criterion2 = nn.SmoothL1Loss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.userDefinedSettings.lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.userDefinedSettings.lr)

    def initialize_value_function_by_expert(self, expert_value_function):
        for target_param, param in zip(self.soft_q_net1.parameters(), expert_value_function.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.soft_q_net2.parameters(), expert_value_function.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net1.parameters(), expert_value_function.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), expert_value_function.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self):
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.userDefinedSettings.soft_update_rate) + param.data * self.userDefinedSettings.soft_update_rate)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.userDefinedSettings.soft_update_rate) + param.data * self.userDefinedSettings.soft_update_rate)

    def hard_update(self):
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

    def update(self, state, action, reward, next_state, done, new_next_action, next_log_prob, alpha, domain_parameter, expert_value_function=None, episode_num=None):
        predict_target_q1 = self.target_soft_q_net1(next_state, new_next_action, domain_parameter)
        predict_target_q2 = self.target_soft_q_net2(next_state, new_next_action, domain_parameter)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - alpha * next_log_prob

        if expert_value_function is not None:
            expert_predict_q = expert_value_function.predict_q_value(next_state, new_next_action, domain_parameter)
            expert_predict_q = expert_predict_q.detach()
            if episode_num is not None:
                alpha = self.userDefinedSettings.expert_value_function_apply_rate
            else:
                alpha = 0.8
            mix_q = (1 - alpha) * target_q_min + alpha * expert_predict_q
            target_q_value = reward + (1 - done) * self.userDefinedSettings.gamma * mix_q
        else:
            target_q_value = reward + (1 - done) * self.userDefinedSettings.gamma * target_q_min

        target_q_value = reward + (1 - done) * self.userDefinedSettings.gamma * target_q_min
        predicted_q_value1 = self.soft_q_net1(state, action, domain_parameter)
        predicted_q_value2 = self.soft_q_net2(state, action, domain_parameter)
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        return q_value_loss1, q_value_loss1, predicted_q_value1, predicted_q_value2

    def predict_q_value(self, state, new_action, domain_parameter):
        predict_q1 = self.soft_q_net1(state, new_action, domain_parameter)
        predict_q2 = self.soft_q_net2(state, new_action, domain_parameter)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        return predicted_new_q_value
