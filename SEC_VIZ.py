# Marching cube -- Learn
# The curvature on the surface
# This file is used for mapping
# VtkPointCloud: the class for adding the points to the vtk object
# vtk_pc: viz single vtk object
# stl_viz: read stl
# stl_change: modify the stl scaling data
# ActorColor: Change the color for the object

import random
import time
import vtk
import numpy as np
from parse import *
from stl import mesh
import SEC_MAP
import SEC_PRE

class vtkGyroCallback():

    def __init__(self):
        pass

    def execute(self, obj, event):
        # Modified segment to accept input for leftCam position
        gyro = (raw_input())
        xyz = parse("{} {} {}", gyro)
        # print xyz
        # "Thread" the renders. Left is called on a right TimerEvent and right is called on a left TimerEvent.
        if obj.ID == 1 and event == 'TimerEvent':
            self.leftCam.SetPosition(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            self.irenL.Render()
            # print "Left"
        elif obj.ID == 2 and event == 'TimerEvent':
            self.rightCam.SetPosition(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            self.irenR.Render()
            # print "Right"
        return

if __name__ == '__main__':

    # test ScanPath
    # test = sec_viz()
    # test.ScanPath()

    # test Double win
    # DoublenWin()

    # test camera
    # CamView()

    # test from mesh to stl
    # test visualization
    test = sec_viz()
    test.test_1()

    # test stl
    # actor_stl = stl_viz()
    # ren = vtk.vtkRenderer()
    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)
    # ren.AddActor(actor_stl)
    # iren.Initialize()
    # renWin.Render()
    # iren.Start()