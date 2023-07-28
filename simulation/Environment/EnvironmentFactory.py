
class EnvironmentFactory():
    def __init__(self, userDefinedSettings):
        self.ENVIRONMENT_NAME = userDefinedSettings.ENVIRONMENT_NAME
        self.userDefinedSettings = userDefinedSettings

    def generate(self, flag=None):

        if self.ENVIRONMENT_NAME == 'Weighing':
            from .Weighing.WeighingEnvironment import WeighingEnvironment
            return WeighingEnvironment(userDefinedSettings=self.userDefinedSettings, parameters=flag)
