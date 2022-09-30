import numpy as np
from UserDefinedSettings import UserDefinedSettings
from Environment.EnvironmentFactory import EnvironmentFactory
from SAC.SACAgent import SACAgent
from Environment.Weighing.WeighingEnvironment import Action_space
import os


class Test:

    def __init__(self):
        self.incline_gain = 0.
        self.policy_type = 'policy'  # naive, policy
        self.action_space = Action_space()

    def root(self):
        averaging_sample_num = 20
        goal_amount_list = [0.005, 0.007, 0.009, 0.010, 0.011, 0.013, 0.015]
        result_list = []

        LEARNING_METHOD = 'SAC'
        userDefinedSettings = UserDefinedSettings(LEARNING_METHOD)
        environmentFactory = EnvironmentFactory(userDefinedSettings)

        env = environmentFactory.generate()

        self.agent = SACAgent(env, userDefinedSettings)

        if self.policy_type == 'policy':
            self.agent.load_model(userDefinedSettings.TEST_DIR)

        for goal_index in range(len(goal_amount_list)):
            for test_num in range(averaging_sample_num):
                state = self.agent.env.reset(reset_info=[goal_amount_list[goal_index], 'goal_condition'])
                final_gap = 0.
                for step_num in range(self.agent.env.MAX_EPISODE_LENGTH):
                    action = self.do_action(state, step_num)
                    next_state, reward, done, _, task_achievement = self.agent.env.step(action, get_task_achievement=True)
                    state = next_state
                    final_gap = task_achievement
                result_list.append([goal_amount_list[goal_index], final_gap])
                print(goal_index, test_num)

        result_list = np.array(result_list)
        np.save(os.path.join(userDefinedSettings.TEST_DIR, 'result_list.npy'), result_list)

    def do_action(self, state, step_num):
        if self.policy_type == 'policy':
            action, _ = self.agent.actor.get_action(state, step=step_num, deterministic=False)
        elif self.policy_type == 'naive':
            action = self.action_from_naive_controller(state)
        return action

    def action_from_naive_controller(self, state):
        current_ball_amount, r_pos, goal_powder_amount = state
        current_ball_amount = current_ball_amount / 500.
        r_pos = r_pos / 30.
        goal_powder_amount = goal_powder_amount / 500.

        ball_amount_gap = current_ball_amount - goal_powder_amount
        incline_gap = -1. * ball_amount_gap

        incline = r_pos + self.incline_gain + incline_gap * 10.
        if ball_amount_gap < 0.:
            shake = 0.
        else:
            shake = ball_amount_gap * 100
        action = np.array([shake, incline])
        self.incline_gain += ball_amount_gap

        action = self.mapping_action(action)
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
