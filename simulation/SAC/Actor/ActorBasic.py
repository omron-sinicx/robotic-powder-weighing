import torch.nn as nn
import torch.optim as optim

from .PolicyNetworkBasic import PolicyNetworkBasic


class ActorBasic(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM, userDefinedSettings):
        super().__init__()

        self.userDefinedSettings = userDefinedSettings
        self.ACTION_DIM = ACTION_DIM
        self.policyNetwork = PolicyNetworkBasic(STATE_DIM, ACTION_DIM, userDefinedSettings)
        self.policy_optimizer = optim.Adam(self.policyNetwork.parameters(), lr=userDefinedSettings.lr)
        self.distillation_loss = nn.MSELoss()

    def initialize_policy(self, target=None):
        if target is not None:
            for target_param, param in zip(self.policyNetwork.parameters(), target.policyNetwork.parameters()):
                target_param.data.copy_(param.data)
        else:
            self.policyNetwork.init_network()

    def evaluate(self, state, get_deterministic_action=False):
        stochastic_action, log_prob, std, deterministic_action = self.policyNetwork.calc_policy(state)
        if get_deterministic_action is True:
            return stochastic_action, log_prob, std, deterministic_action
        else:
            return stochastic_action, log_prob, std

    def get_action(self, state, step=None, deterministic=True, random_action_flag=False, agent_id=None, get_std=False):
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

        if get_std:
            return execute_action, method_depend_info, std
        else:
            return execute_action, method_depend_info

    def update(self, alpha, log_prob, q_value):
        policy_loss = (alpha * log_prob - q_value).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss
