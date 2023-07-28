#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import time
import pygame
from ...simulation.Environment.Weighing.control_joy import JoyController

PC_IP_ADDRESS = "XXX.XXX.XXX.XXX"
ROBOT_IP_ADDRESS = "YYY.YYY.YYY.YYY"


class UR_EE_Control:
    def __init__(self):
        self.config_ur_socket()
        self.config_joy()

    def config_ur_socket(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        PORT = 30002
        self.s.connect((ROBOT_IP_ADDRESS, PORT))

    def config_joy(self):
        pygame.init()
        self.joy_controller = JoyController(0)
        self.scale = 0.1
        self.action = [0.00] * 6

    def toBytes(self, str):
        return bytes(str.encode())

    def move(self, action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        x = action[0]
        y = action[1]
        z = action[2]

        acc = 0.50
        vel = 0.10

        self.s.send(
            self.toBytes(
                "def myProg():"
                + "\n"
                + "begin_pos = get_actual_tcp_pose()"  # Get current pose
                + "\n"
                + "pos_end = pose_add(begin_pos, p["
                + str(-y)  # transformed
                + ", "
                + str(-x)  # transformed
                + ", "
                + str(z)
                + ", "
                + "0.0"
                + ", "
                + "0.0"
                + ", "
                + "0.0"
                + "])"
                + "\n"
                + "movel(pos_end , a="
                + str(acc)
                + ", v="
                + str(vel)
                + ")"
                + "\n"
                + "end"
                + "\n"
            )
        )

    def get_joy_input(self):
        eventlist = pygame.event.get()
        self.joy_controller.get_controller_value(eventlist)
        x = -self.joy_controller.l_hand_y / self.scale
        y = self.joy_controller.l_hand_x / self.scale
        z = -self.joy_controller.r_hand_y / self.scale
        self.action[0] = x
        self.action[1] = y
        self.action[2] = z
        print(self.action)

    def main(self):
        step = 0
        while True:
            self.get_joy_input()
            self.move(self.action)
            step += 1
            time.sleep(1.0)
            print("time step: ", step)


if __name__ == "__main__":
    ur_ee_control = UR_EE_Control()
    ur_ee_control.main()
