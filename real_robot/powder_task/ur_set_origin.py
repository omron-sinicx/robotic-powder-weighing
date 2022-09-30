#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import time

HOST = "192.168.11.5"
PORT = 30002


def toBytes(str):
    return bytes(str.encode())


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send(toBytes("movej([0,-1.57,1.57,-1.57,-1.57,0], a=1.40, v=0.5)" + "\n"))
    # ラジアン
    time.sleep(10)
    data = s.recv(1024)
    s.close()
    print("Receved", repr(data))
    print("Program finish")


if __name__ == "__main__":
    main()
