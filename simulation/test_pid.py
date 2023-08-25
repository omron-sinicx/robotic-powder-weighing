import numpy as np
from UserDefinedSettings import UserDefinedSettings
from Environment.EnvironmentFactory import EnvironmentFactory
from SAC.SACAgent import SACAgent
from Environment.Weighing.WeighingEnvironment import Action_space
import os


class Test:

    def __init__(self):
        self.action_space = Action_space()
        pid_gain_list = np.load('pid_gain.npy')
        self.pid_incline = pid_gain_list[0]
        self.pid_shake = pid_gain_list[1]
        self.e = 0.
        self.e1 = 0.
        self.es = 0.

    def root(self):
        averaging_sample_num = 1
        goal_amount_list = [0.005, 0.007, 0.009, 0.010, 0.011, 0.013, 0.015]
        result_list = []

        LEARNING_METHOD = 'SAC'
        userDefinedSettings = UserDefinedSettings(LEARNING_METHOD)
        environmentFactory = EnvironmentFactory(userDefinedSettings)

        env = environmentFactory.generate()

        self.agent = SACAgent(env, userDefinedSettings)

        for goal_index in range(len(goal_amount_list)):
            for test_num in range(averaging_sample_num):
                state = self.agent.env.reset(reset_info=[goal_amount_list[goal_index], 'goal_condition'])
                final_gap = 0.
                for step_num in range(self.agent.env.MAX_EPISODE_LENGTH):
                    action = self.action_from_naive_controller(state)
                    next_state, reward, done, _, task_achievement = self.agent.env.step(action, get_task_achievement=True)
                    state = next_state
                    final_gap = task_achievement
                result_list.append(final_gap)
                print(goal_index, test_num)

        result_list = np.array(result_list)
        np.save(os.path.join('result_list.npy'), result_list)

    def action_from_naive_controller(self, state):
        current_ball_amount, _, goal_powder_amount = state
        current_ball_amount = current_ball_amount / 500.
        goal_powder_amount = goal_powder_amount / 500.

        self.e1 = self.e
        self.e = current_ball_amount - goal_powder_amount  # 偏差（e） = 目的値（goal_powder_amount） - 前回の実現値
        self.es += self.e

        incline = self.pid_incline[0] * self.e + self.pid_incline[1] * self.es + self.pid_incline[2] * (self.e - self.e1)
        shake = self.pid_shake[0] * self.e + self.pid_shake[1] * self.es + self.pid_shake[2] * (self.e - self.e1)

        if self.e < 0.:
            shake = 0.
            incline = 0.

        action = np.array([incline, shake])
        return action

    def mapping_action(self, action):
        """
        a~bを-1~1に変換
        """
        action = -1. + 2 * (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        action = np.clip(action, -1., 1.)
        return action


if __name__ == '__main__':
    test = Test()
    test.root()
