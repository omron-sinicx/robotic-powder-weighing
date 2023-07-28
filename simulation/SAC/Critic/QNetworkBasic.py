import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Normal
# import torch.optim as optim
# import numpy as np
from .initialize import linear_weights_init


class QNetworkBasic(nn.Module):

    def __init__(self, state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings):
        super().__init__()

        self.use_domain_flag = userDefinedSettings.DOMAIN_RANDOMIZATION_FLAG
        HIDDEN_NUM = userDefinedSettings.HIDDEN_NUM

        if self.use_domain_flag:
            input_dim = state_dim + action_dim + DOMAIN_PARAMETER_DIM
        else:
            input_dim = state_dim + action_dim

        self.linear1 = nn.Linear(input_dim, HIDDEN_NUM)
        self.linear2 = nn.Linear(HIDDEN_NUM, HIDDEN_NUM)
        self.linear3 = nn.Linear(HIDDEN_NUM, 1)
        self.linear3.apply(linear_weights_init)

    def forward(self, state, action, domain_parameter):
        if self.use_domain_flag:
            inputs = [state, action, domain_parameter]
        else:
            inputs = [state, action]

        x = torch.cat(inputs, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value
