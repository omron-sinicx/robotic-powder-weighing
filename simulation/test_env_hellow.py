from Environment.Weighing.WeighingEnvironment import WeighingEnvironment
from UserDefinedSettings import UserDefinedSettings

import numpy as np


def test():
    userDefinedSettings = UserDefinedSettings()
    weighingEnvironment = WeighingEnvironment(userDefinedSettings=userDefinedSettings)
    for _ in range(10):
        weighingEnvironment.reset()
        for _ in range(3):
            action = np.array([0, -0.5])
            next_state, reward, done, domain_parameter = weighingEnvironment.step(action)


if __name__ == "__main__":
    test()
