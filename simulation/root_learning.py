import subprocess
from UserDefinedSettings import UserDefinedSettings


def root():
    userDefinedSettings = UserDefinedSettings()
    episode_num_per_reset = userDefinedSettings.learning_episode_num

    flag = ''.join(userDefinedSettings.flag)
    total_roop_num = int(userDefinedSettings.total_episode_num / episode_num_per_reset)
    for current_loop_num in range(total_roop_num):
        if userDefinedSettings.RENDER_FLAG:
            command = ["python3", "root_sac_learning.py", "--flag", flag, "--episode", str(current_loop_num * episode_num_per_reset), "--render"]
        else:
            command = ["python3", "root_sac_learning.py", "--flag", flag, "--episode", str(current_loop_num * episode_num_per_reset)]
        subprocess.run(command)


if __name__ == '__main__':
    root()
