from Environment.Weighing.WeighingEnvironment import WeighingEnvironment
from UserDefinedSettings import UserDefinedSettings


def test():
    userDefinedSettings = UserDefinedSettings()
    weighingEnvironment = WeighingEnvironment(userDefinedSettings=userDefinedSettings)
    weighingEnvironment.joy_loop()


if __name__ == "__main__":
    test()
