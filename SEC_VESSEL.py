# This function is used for object recognition
# 1: Color segmentation
# 2. Vessel based segmentation

# color segmentation + edge detection + circle object detection
# Algorithm:
# determine the color type -- use color histogram
# Edge detection --
# circle object detection -- focus on the image

import vtk
import pcl
import cv2

filename = "/home/maguangshen/PycharmProjects/BTL_GS/Data/vessel.stl"

reader = vtk.vtkSTLReader()
reader.SetFileName(filename)

mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(reader.GetOutput())
else:
    mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()

