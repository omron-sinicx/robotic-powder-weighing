import time
import pygame
from pygame.locals import *


def main():
    pygame.init()
    controller = JoyController(0)
    while True:
        eventlist = pygame.event.get()
        controller.get_controller_value(eventlist)
        time.sleep(0.1)


class JoyController:
    def __init__(self, id):
        pygame.joystick.init()
        # print("num con: ", pygame.joystick.get_count())
        self.joy = pygame.joystick.Joystick(id)

        self.l_hand_x = 0.0
        self.l_hand_y = 0.0
        self.r_hand_x = 0.0
        self.r_hand_y = 0.0
        self.L2 = 0.0
        self.R2 = 0.0

        self.button_A = False
        self.button_B = False
        self.button_X = False
        self.button_Y = False

        self.button_UD = False
        self.button_RL = False
        self.button_R1 = False
        self.button_R2 = False

        self.button_L1 = False
        self.button_L2 = False

        self.button_BACK = False
        self.button_START = False

        self.cross_key = [0, 0]

    def get_controller_value(self, event):
        for eventlist in event:
            # Read stick
            if eventlist.type == JOYAXISMOTION:
                self.l_hand_x = self.joy.get_axis(0)
                self.l_hand_y = self.joy.get_axis(1)
                self.r_hand_x = self.joy.get_axis(3)
                self.r_hand_y = self.joy.get_axis(4)
                self.L2 = (self.joy.get_axis(2) + 1.0) / 2.0
                self.R2 = (self.joy.get_axis(5) + 1.0) / 2.0

            # Read button
            self.button_A = self.joy.get_button(0)
            self.button_B = self.joy.get_button(1)
            self.button_X = self.joy.get_button(2)
            self.button_Y = self.joy.get_button(3)
            self.button_L1 = self.joy.get_button(4)
            self.button_R1 = self.joy.get_button(5)
            self.button_BACK = self.joy.get_button(6)
            self.button_START = self.joy.get_button(7)

            # Read hat
            if eventlist.type == JOYHATMOTION:
                self.cross_key = self.joy.get_hat(0)

            # print(
            #     self.L2,
            #     self.R2,
            #     self.l_hand_x,
            #     self.l_hand_y,
            #     self.r_hand_x,
            #     self.r_hand_y,
            #     self.cross_key[0],
            #     self.cross_key[1],
            #     # self.joy.get_button(0),
            #     # self.joy.get_button(1),
            #     # self.joy.get_button(2),
            #     # self.joy.get_button(3),
            #     # self.joy.get_button(4),
            #     # self.joy.get_button(5),
            #     # self.joy.get_button(6),
            #     # self.joy.get_button(7),
            #     # self.joy.get_button(8),
            #     # self.joy.get_button(9),
            # )


if __name__ == "__main__":
    main()
