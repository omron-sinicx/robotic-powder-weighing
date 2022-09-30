#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import socket
import pygame
import sys
import numpy as np
import torch

from ScaleITX120Interface import ScaleITX120Interface
from URPositionController import URPositionController
from UserDefinedSettings import UserDefinedSettings  # noqa

sys.path.append("../../Nextcloud/code/deepRL/distillation_proposed")  # noqa
from SAC.Actor.ActorLSTM import ActorLSTM  # noqa


class PowderEnvironmentController:
    def __init__(self):
        self.config_ur_socket()
        self.config_scale()
        self.config_position_control()

        self.userDefinedSettings = UserDefinedSettings()
        self.actor = ActorLSTM(STATE_DIM=3, ACTION_DIM=2, userDefinedSettings=self.userDefinedSettings)
        self.load_model(self.userDefinedSettings.TEST_DIR)

        self.action_type = 'relative'
        self.action_space = Action_space(action_type=self.action_type)

    def load_model(self, path):
        self.actor.policyNetwork.load_state_dict(torch.load(path, map_location=torch.device(self.userDefinedSettings.DEVICE)))
        self.actor.policyNetwork.eval()

    def config_camera(self):
        from RealsenseController_AddRGBImage import RealsenseController
        self.realsenseController = RealsenseController()

    def config_position_control(self):
        self.positionController = URPositionController()

    def config_scale(self):
        self.scale = ScaleITX120Interface(port="/dev/ttyUSB0")

    def config_ur_socket(self):
        self.config_server()
        self.HOST = "192.168.11.2"  # remote host
        self.PORT = 30002  # UR port

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.HOST, self.PORT))
        self.s.listen(5)
        self.c, self.addr = self.s.accept()

    def config_server(self):
        with open("./move_ee_without_force.script", "r") as file:
            prog = file.read()

        HOST = "192.168.11.5"
        PORT = 30001

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.send(self.toBytes(prog))

    def toBytes(self, str):
        return bytes(str.encode())

    def toStr(self, byte_data):
        return str(byte_data.decode())

    def get_state(self):
        msg = self.c.recv(1024)
        msg = self.toStr(msg)
        msg = msg.replace("p", "")
        state_list = msg.split("_")
        state_dict = {
            "ee_pose": eval(state_list[0]),
            "joints": eval(state_list[1]),
            # "force": eval(state_list[2]),
        }
        return state_dict

    def send_action(self, action):
        self.c.send(
            self.toBytes(
                "("
                + str(action[0])
                + ","
                + str(action[1])
                + ","
                + str(action[2])
                + ","
                + str(action[3])
                + ","
                + str(action[4])
                + ","
                + str(action[5])
                + ")"
            )
        )

    def move_init_position(self):
        self.c.close()
        self.s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        HOST = "192.168.11.5"
        PORT = 30002
        s.connect((HOST, PORT))
        base = -2.46
        shoulder = -70.46
        elbow = 90.74
        wrist1 = -110.46
        wrist2 = -142.48
        wrist3 = 358.25
        c = np.pi / 180.
        move_txt = '[' + str(base * c) + ',' + str(shoulder * c) + ',' + str(elbow * c) + ',' + str(wrist1 * c) + ',' + str(wrist2 * c) + ',' + str(wrist3 * c) + ']'
        s.send(self.toBytes("movej(" + move_txt + ',' + "a=1.40, v=0.5)" + "\n"))
        time.sleep(5)
        s.close()
        self.config_ur_socket()

    def move_position_control(self, action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocity=0.1, timer=1., config_flag=True):
        """
        x
        y
        z:上がプラス
        rx:ロボット方向
        ry:すくう方向がマイナス
        rz
        """
        self.c.close()
        self.s.close()
        self.positionController.move(action, velocity=velocity)
        time.sleep(timer)
        self.config_ur_socket()

    def get_powder(self):
        print('init for act')
        self.move_init_position()
        print('start act')
        trajectory_list = np.load('trajectory_get_powder.npy')
        for action in trajectory_list:
            self.send_action(action)
            time.sleep(0.01)

    def stir_powder(self):
        print('init for act')
        self.move_init_position()
        print('start act')
        trajectory_list = np.load('trajectory_stir.npy')
        for action in trajectory_list:
            self.send_action(action)
            time.sleep(0.01)

    def get_ee_pose(self):
        self.config_ur_socket()
        state_dict = self.get_state()
        ee_pose = state_dict['ee_pose']
        print(np.array(ee_pose[4]) * (180. / np.pi))
        return ee_pose[4]

    def get_image(self):
        rgb_image = self.realsenseController.get_rgb_image()
        return rgb_image

    def get_scale(self):
        return -0.001 * self.scale.readValue()  # [g]

    def calc_reward(self):
        self.current_ball_amount = self.get_scale()
        target_ball_amount = 0.005  # [g]
        reward_scaling = 1000.
        reward = -np.abs(self.current_ball_amount - target_ball_amount) * reward_scaling
        return reward, self.current_ball_amount

    def get_action(self, state, step_num):
        action, _ = self.actor.get_action(state, step=step_num, deterministic=True)
        action = self.mapping_action(action)
        action_x, action_r = action
        return action_x, action_r

    def mapping_action(self, action):
        """
        入力される行動の範囲が-1から+1であることを仮定
        """
        assert (action.any() >= -1) and (action.any() <= 1), 'expected actions are \"-1 to +1\". input actions are {}'.format(action)
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)  # (-1,1) -> (low,high)
        action = np.clip(action, low, high)  # (-X,+Y) -> (low,high)
        return action

    def shake_spoon(self, action_x):
        distance = 0.003
        velocity_scale = 0.1
        push_time = 0.06
        self.c.close()
        self.s.close()
        self.positionController.move([distance * action_x, 0, 0, 0, 0, 0], velocity=action_x * velocity_scale)
        time.sleep(push_time)
        self.positionController.move([-distance * 0.3 * action_x, 0, 0, 0, 0, 0], velocity=action_x * velocity_scale)
        time.sleep(1)
        self.config_ur_socket()

    def incline_spoon(self, action_r):
        """
        rは落とす方向がマイナスで初期が0
        """
        action_weight = 1.
        self.move_position_control(action=[0, 0, 0, 0, action_r * action_weight, 0], velocity=0.02, timer=5.)

    def get_observation(self):
        state = np.array([self.current_ball_amount * 500., self.current_rad * 30., self.goal_powder_amount * 500.]).reshape(-1)
        return state

    def do_experiments(self):
        """
        スプーンの粉を落とす方向がマイナスで単位はrad
        """
        initial_rad = -10. * np.pi / 180.
        total_step_num = 6  # 10
        self.goal_powder_amount = 0.005  # 0.005
        inline_flag = True
        shake_flag = True

        self.move_init_position()
        self.stir_powder()
        self.get_powder()
        self.move_position_control(action=[0, 0, 0.03, 0, 0, 0])  # go to camera position
        self.move_position_control(action=[-0.007, 0, 0, 0, 0, 0])  # go to camera position
        self.go_to_certain_position()  # go to 10 degree

        self.current_rad = initial_rad
        reward, self.current_ball_amount = self.calc_reward()
        state = self.get_observation()
        total_reward = 0.
        for step in range(total_step_num):
            action_x, action_r = self.get_action(state, step)
            if inline_flag:
                if self.action_type == 'absolute':
                    bias = 0.
                    self.incline_spoon(action_r - self.current_rad + bias)  # 絶対座標で入力
                    self.current_rad = action_r
                else:
                    if self.current_rad + action_r > -10. * np.pi / 180.:
                        print('action', action_r * 180. / np.pi, 'low')
                        action_r = -10. * np.pi / 180. - self.current_rad
                        self.current_rad = -10. * np.pi / 180.
                    elif self.current_rad + action_r < -30. * np.pi / 180.:
                        print('action', action_r * 180. / np.pi, 'high')
                        action_r = -30. * np.pi / 180. - self.current_rad
                        self.current_rad = -30. * np.pi / 180.
                    else:
                        self.current_rad += action_r
                        print('action', action_r * 180. / np.pi, 'range')
                    self.incline_spoon(action_r)  # 相対座標で入力
            print('degree', self.current_rad * 180. / np.pi)

            if shake_flag:
                self.shake_spoon(action_x)
            reward, self.current_ball_amount = self.calc_reward()

            state = self.get_observation()

            total_reward += reward
            print('x:{:.3f} | r:{:.3f}'.format(action_x, action_r))
        print('total_reward:', total_reward)

        self.move_position_control(action=[10, 0, 0, 0, 0, 0], velocity=0.02, timer=6.1)  # go to camera position
        self.move_position_control(action=[0.1, 0, 0, 0, -90. * np.pi / 180., 0], velocity=0.1, timer=3)  # go to 10 degree

    def do_test(self):
        """
        スプーンの粉を落とす方向がマイナスで単位はrad
        """
        # self.move_init_position()
        self.get_ee_pose()
        sys.exit()
        for i in range(3):
            self.move_position_control(action=[0, 0, 0, 0, 5 * np.pi / 180., 0], velocity=0.01, timer=10.)  # go to 10 degree
            self.get_ee_pose()

    def go_to_certain_position(self):
        target = -75. * np.pi / 180.  # 10 degree
        gap = 0.3 * np.pi / 180.   # 0.3 degree

        while(True):
            current = self.get_ee_pose()
            move = target - current
            print(move)
            if np.abs(move) > gap:
                self.move_position_control(action=[0, 0, 0, 0, move, 0], velocity=0.02, timer=2.)  # go to 10 degree
            else:
                break

    def shake_test(self):
        for _ in range(5):
            self.shake_spoon(0.1)
            self.shake_spoon(0.5)
            self.shake_spoon(1)


class Action_space:
    def __init__(self, action_type='absolute'):
        if action_type == 'absolute':
            self.low = np.array([0., -30. * np.pi / 180.])
            self.high = np.array([1., -10. * np.pi / 180.])
        else:
            self.low = np.array([0., -3. * np.pi / 180.])
            self.high = np.array([1., 3. * np.pi / 180.])
        self.ACTION_DIM = len(self.high)

    def sample(self):
        # 行動は-1から+1であることを仮定
        action = np.random.rand()
        action = 2. * (action - 0.5)
        return action


if __name__ == "__main__":
    powderEnvironmentController = PowderEnvironmentController()
    powderEnvironmentController.do_experiments()
