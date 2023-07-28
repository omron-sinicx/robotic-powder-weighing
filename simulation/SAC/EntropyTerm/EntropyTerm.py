import torch
from torch.optim import Adam


class EntropyTerm(object):

    def __init__(self, ACTION_DIM, userDefinedSettings):
        self.entropy_tuning = userDefinedSettings.entropy_tuning

        if self.entropy_tuning:
            target_entropy = ACTION_DIM * userDefinedSettings.entropy_tuning_scale
            target_entropy = (target_entropy, )
            self.target_entropy = -torch.prod(torch.Tensor(target_entropy).to(userDefinedSettings.DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=userDefinedSettings.DEVICE)
            self.alpha = self.log_alpha.exp()
            self.optimizer = Adam([self.log_alpha], lr=userDefinedSettings.lr)
        else:
            self.alpha = torch.tensor(userDefinedSettings.entropy_coefficient).to(userDefinedSettings.DEVICE)

    def calc_entropy_loss(self, entropy):
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach())
        return entropy_loss

    def update(self, entropies):
        if self.entropy_tuning is True:
            entropy_loss = self.calc_entropy_loss(entropies)
            self.optimizer.zero_grad()
            entropy_loss.backward()
            self.optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            entropy_loss = 0.
        return entropy_loss
