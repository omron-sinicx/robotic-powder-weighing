import torch
import torch.nn as nn
import torch.optim as optim

from .PolicyNetworkLSTM import PolicyNetworkLSTM
from .ActorBasic import ActorBasic


class ActorLSTM(ActorBasic):
    def __init__(self, STATE_DIM, ACTION_DIM, userDefinedSettings):
        super(ActorLSTM, self).__init__(STATE_DIM, ACTION_DIM, userDefinedSettings)
        self.policyNetwork = PolicyNetworkLSTM(STATE_DIM, ACTION_DIM, userDefinedSettings)
        self.policy_optimizer = optim.Adam(self.policyNetwork.parameters(), lr=userDefinedSettings.lr)
        self.distillation_loss = nn.MSELoss()

    def evaluate(self, state, last_action, hidden_in, get_deterministic_action=False):
        action, log_prob, std, deterministic_action, hidden_out = self.policyNetwork.calc_policy(state, last_action, hidden_in)
        if get_deterministic_action is True:
            return action, log_prob, std, deterministic_action
        else:
            return action, log_prob, std

    def get_action(self, state, step=None, deterministic=True, random_action_flag=False):
        if step == 0:
            self.reset_policy_parameters()
        state = self.policyNetwork.format_numpy2torch(state)
        hidden_in = self.current_hidden_input
        last_action = self.current_last_action

        action, log_prob, std, deterministic_action, hidden_out = self.policyNetwork.calc_policy(state, last_action, hidden_in)
        if deterministic:
            action = deterministic_action
        else:
            action = action

        self.current_hidden_input = hidden_out
        self.current_last_action = action
        last_action_in_memory = self.policyNetwork.format_torch2numpy(last_action)
        lstm_info = {'hidden_in': hidden_in, 'hidden_out': hidden_out, 'last_action': last_action_in_memory}

        if random_action_flag:
            execute_action = self.policyNetwork.sample_action(format='numpy')
        else:
            execute_action = self.policyNetwork.format_torch2numpy(action)

        return execute_action, lstm_info

    def reset_policy_parameters(self):
        self.initial_hidden_input = (torch.zeros([1, 1, self.userDefinedSettings.HIDDEN_NUM], dtype=torch.float).to(self.userDefinedSettings.DEVICE),
                                     torch.zeros([1, 1, self.userDefinedSettings.HIDDEN_NUM], dtype=torch.float).to(self.userDefinedSettings.DEVICE))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        self.current_hidden_input = self.initial_hidden_input
        self.initial_hidden_output = None
        action = self.policyNetwork.sample_action(format='numpy')
        self.initial_last_action = self.policyNetwork.format_numpy2torch(action)  # numpy[1] -> torch[1,1,1]

        self.current_last_action = self.initial_last_action

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

    def DistRL_update(self, state, last_action, hidden_in, target_action):
        mean, log_std, hidden_out = self.policyNetwork.forward(state, last_action, hidden_in)
        student_action = torch.tanh(mean)
        loss = self.distillation_loss(student_action, target_action)
        self.policy_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        return loss

    def DistRL_update_std(self, state, last_action, hidden_in, target_log_std):
        mean, log_std, hidden_out = self.policyNetwork.forward(state, last_action, hidden_in)
        loss = self.distillation_loss(log_std, target_log_std)
        self.policy_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        return loss
