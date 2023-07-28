from collections import deque
import numpy as np


class TotalRewardService:

    def __init__(self, userDefinedSettings):
        self.trainPeakChecker = ProgressChecker(userDefinedSettings)
        self.testPeakChecker = ProgressChecker(userDefinedSettings)


class ProgressChecker(object):
    def __init__(self, userDefinedSettings=None, length=None):
        self.userDefinedSettings = userDefinedSettings
        if length is not None:
            self.latest_value_queue = deque(maxlen=length)
        else:
            self.latest_value_queue = deque(maxlen=userDefinedSettings.average_num_for_model_save)
        self.max_averaged_value = -999999.9

    def append_and_check(self, value):
        self.append(value)
        return self.check_peak()

    def append_and_value(self, value):
        self.append(value)
        current_value = np.mean(self.latest_value_queue)
        return current_value

    def append(self, value):
        self.latest_value_queue.append(value)

    def get_mean(self):
        current_value = np.mean(self.latest_value_queue)
        return current_value

    def check_peak(self):
        current_value = np.median(self.latest_value_queue)
        if current_value > self.max_averaged_value:
            self.max_averaged_value = current_value
            return True
        else:
            return False
