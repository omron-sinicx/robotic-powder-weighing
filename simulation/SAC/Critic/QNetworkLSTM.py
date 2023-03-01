import torch
import torch.nn as nn
import torch.nn.functional as F
from .initialize import linear_weights_init


class QNetworkLSTM(nn.Module):

    def __init__(self, state_dim, action_dim, DOMAIN_PARAMETER_DIM, userDefinedSettings):
        super().__init__()

        self.use_domain_flag = userDefinedSettings.DOMAIN_RANDOMIZATION_FLAG
        HIDDEN_NUM = userDefinedSettings.HIDDEN_NUM

        if self.use_domain_flag:
            input_dim = state_dim + action_dim + DOMAIN_PARAMETER_DIM
        else:
            input_dim = state_dim + action_dim

        self.linear1 = nn.Linear(input_dim, HIDDEN_NUM)
        self.linear2 = nn.Linear(state_dim + action_dim, HIDDEN_NUM)
        self.lstm1 = nn.LSTM(HIDDEN_NUM, HIDDEN_NUM, batch_first=True)
        self.linear3 = nn.Linear(2 * HIDDEN_NUM, HIDDEN_NUM)
        self.linear4 = nn.Linear(HIDDEN_NUM, 1)
        self.linear4.apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in, domain_parameter):
        if self.use_domain_flag:
            inputs = [state, action, domain_parameter]
        else:
            inputs = [state, action]

        # branch 1
        fc_branch = torch.cat(inputs, -1)
        fc_branch = F.relu(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = F.relu(self.linear3(merged_branch))
        x = self.linear4(x)

        return x, lstm_hidden
