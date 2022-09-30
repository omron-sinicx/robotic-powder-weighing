# -*- coding: utf-8 -*-
import xml.etree.ElementTree as gfg
import numpy as np


class BallGenerator(object):

    def __init__(self):
        pass

    def generate(self, file_name=None, ball_radius=None, ball_mass=None):

        root = gfg.Element("robot", name="ball")

        ###
        link = gfg.Element("link", name="ball")
        root.append(link)
        visual = gfg.Element("visual")
        link.append(visual)
        gfg.SubElement(visual, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        visual.append(geometry)
        gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        collision = gfg.Element("collision")
        link.append(collision)
        gfg.SubElement(collision, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        collision.append(geometry)
        gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        inertial = gfg.Element("inertial")
        link.append(inertial)
        gfg.SubElement(inertial, "mass", value=str(ball_mass * 10. / 3.))  # fix scale
        # gfg.SubElement(inertial, "mass", value="0.0001")
        gfg.SubElement(inertial, "inertia", ixx="0.01", ixy="0.0", ixz="0.0", iyy="0.01", iyz="0.0", izz="0.01")

        if True:
            self.make_file(file_name, root)
        else:
            return gfg.dump(root)

    def make_file(self, file_name, root):
        # ElementTreeを追加
        tree = gfg.ElementTree(root)
        # ファイルの書き込み
        with open(file_name, "wb") as files:
            tree.write(files)


# コードを実行
if __name__ == "__main__":
    urdfGenerator = BallGenerator()
    urdfGenerator.generate(file_name="Environment/Weighing/BallHLS.urdf", ball_radius=0.01, ball_mass=0.001)
