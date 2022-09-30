from abc import ABCMeta, abstractmethod


class InterfaceEnvironment(metaclass=ABCMeta):

    @abstractmethod
    def get_state_action_space(self):
        STATE_DIM = self.env.observation_space.shape
        ACTION_DIM = self.env.action_space.shape
        return STATE_DIM, ACTION_DIM

    @abstractmethod
    def reset(self):
        state = self.env.reset()
        return state

    @abstractmethod
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    @abstractmethod
    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    @abstractmethod
    def random_action_sample(self):
        action = self.env.action_space.sample()
        return action

    @abstractmethod
    def render(self):
        self.env.render()

    @abstractmethod
    def __del__(self):
        self.env.close()
