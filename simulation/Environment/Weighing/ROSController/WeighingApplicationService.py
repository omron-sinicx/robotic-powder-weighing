import sys
import numpy as np

from .Talker import Talker

from ..WeighingEnvironment import WeighingEnvironment  # noqa


"""
Usage:
rosmaster
python listener.py
"""


class WeighingApplicationService:
    def __init__(self, userDefinedSettings=None, domain_range=None):
        self.talker = Talker(userDefinedSettings=userDefinedSettings)

        self.weighingEnvironment = WeighingEnvironment(userDefinedSettings=userDefinedSettings, create_env_flag=False)
        self.DOMAIN_PARAMETER_DIM = self.weighingEnvironment.DOMAIN_PARAMETER_DIM
        self.MAX_EPISODE_LENGTH = self.weighingEnvironment.MAX_EPISODE_LENGTH
        self.STATE_DIM = self.weighingEnvironment.STATE_DIM
        self.ACTION_DIM = self.weighingEnvironment.ACTION_DIM
        self.step_num = 0

    def reset(self, reset_info=[-999., 'goal_condition']):
        self.step_num = 0
        self.talker.make()
        self.talker.port2env()
        state = self.talker.reset(np.array([reset_info[0]]))
        return state

    def step(self, action, get_task_achievement=False):
        next_state, reward, done, domain_parameter, task_achievement = self.talker.step(action)

        if self.step_num == self.MAX_EPISODE_LENGTH - 1:
            self.talker.delete()
        else:
            self.step_num += 1
        if get_task_achievement is True:
            return next_state, reward, done, domain_parameter, task_achievement
        else:
            return next_state, reward, done, domain_parameter

    def render(self):
        pass


if __name__ == '__main__':
    weighingApplicationService = WeighingApplicationService()

    for i in range(1):
        print(i)
        state = weighingApplicationService.reset()
        print('state', state)
        for j in range(10):
            action = np.array([0, 0])
            next_state, reward, done, domain_parameter, task_achievement = weighingApplicationService.step(action, get_task_achievement=True)
            print('reward', reward)
