#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket

PC_IP_ADDRESS = "XXX.XXX.XXX.XXX"
ROBOT_IP_ADDRESS = "YYY.YYY.YYY.YYY"


class URPositionController:
    def __init__(self):
        self.config_ur_socket()

    def config_ur_socket(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        PORT = 30003
        self.s.connect((ROBOT_IP_ADDRESS, PORT))

    def toBytes(self, str):
        return bytes(str.encode())

    def move(self, action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocity=None):
        x = action[0]
        y = action[1]
        z = action[2]
        rx = action[3]
        ry = action[4]
        rz = action[5]

        acc = 1.  # 0.50
        if velocity is None:
            vel = 0.10
        else:
            vel = velocity

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
                + str(rx)
                + ", "
                + str(ry)
                + ", "
                + str(rz)
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

    def main(self):
        self.move(action=[0, 0, 0.001, 0, 0, 0])


if __name__ == "__main__":
    ur_ee_control = URPositionController()
    ur_ee_control.main()
