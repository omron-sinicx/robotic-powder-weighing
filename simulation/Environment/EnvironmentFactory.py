
class EnvironmentFactory():
    def __init__(self, userDefinedSettings):
        self.ENVIRONMENT_NAME = userDefinedSettings.ENVIRONMENT_NAME
        self.userDefinedSettings = userDefinedSettings

    def generate(self, domain_range=None):
        if self.ENVIRONMENT_NAME == 'Weighing':
            if self.userDefinedSettings.ros:
                from .Weighing.ROSController.WeighingApplicationService import WeighingApplicationService
                return WeighingApplicationService(userDefinedSettings=self.userDefinedSettings, domain_range=domain_range)
            else:
                from .Weighing.WeighingEnvironment import WeighingEnvironment
                return WeighingEnvironment(userDefinedSettings=self.userDefinedSettings, domain_range=domain_range)
