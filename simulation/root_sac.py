from UserDefinedSettings import UserDefinedSettings
from Environment.EnvironmentFactory import EnvironmentFactory
from SAC.SACAgent import SACAgent
from LearningCommonParts.ItemDebugHandler import ItemDebugHandler


def root():

    LEARNING_METHOD = 'SAC'
    userDefinedSettings = UserDefinedSettings(LEARNING_METHOD)
    environmentFactory = EnvironmentFactory(userDefinedSettings)
    itemDebugHandler = ItemDebugHandler(path=userDefinedSettings.LOG_DIRECTORY)

    env = environmentFactory.generate()
    agent = SACAgent(env, userDefinedSettings)
    agent.itemDebugHandler = itemDebugHandler

    if userDefinedSettings.TEST_FLAG:
        agent.test(model_path=userDefinedSettings.TEST_DIR, policy=None, domain_num=None, test_num=5, render_flag=True, reward_show_flag=True)
    else:
        agent.train()


if __name__ == '__main__':
    root()
