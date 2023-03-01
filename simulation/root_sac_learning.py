from UserDefinedSettings import UserDefinedSettings
from Environment.EnvironmentFactory import EnvironmentFactory
from SAC.SACAgent import SACAgent


def root():
    userDefinedSettings = UserDefinedSettings()
    environmentFactory = EnvironmentFactory(userDefinedSettings)

    flag = userDefinedSettings.flag

    env = environmentFactory.generate(flag)
    agent = SACAgent(env, userDefinedSettings)
    agent.train()


if __name__ == '__main__':
    root()
