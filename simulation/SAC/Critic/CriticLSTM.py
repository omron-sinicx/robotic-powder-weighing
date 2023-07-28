import torch

from .CriticBasic import CriticBasic
from .QNetworkLSTM import QNetworkLSTM


class CriticLSTM(CriticBasic):

    def __init__(self, state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings):
        super().__init__(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings)
        self.value_lstm_flag = True
        self.action_dim = action_dim
        self.userDefinedSettings = userDefinedSettings
        Network = QNetworkLSTM
        self.soft_q_net1 = Network(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.soft_q_net2 = Network(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.target_soft_q_net1 = Network(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.target_soft_q_net2 = Network(state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings).to(userDefinedSettings.DEVICE)
        self.network_initialization()

    def update(self, state, action, reward, next_state, done, lstm_term, new_next_action, next_log_prob, alpha, domain_parameter, actor=None, episode_num=None):

        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, lstm_term['hidden_out'], domain_parameter)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, lstm_term['hidden_out'], domain_parameter)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.userDefinedSettings.gamma * target_q_min
        predicted_q_value1, _ = self.soft_q_net1(state, action, lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        predicted_q_value2, _ = self.soft_q_net2(state, action, lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        return q_value_loss1, q_value_loss1, predicted_q_value1, predicted_q_value2

    def predict_q_value(self, state, new_action, last_action, hidden_in, domain_parameter):
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in, domain_parameter)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in, domain_parameter)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        return predicted_new_q_value
