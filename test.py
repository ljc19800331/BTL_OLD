# Goals: Capture multiple viewpoint and get the RGBD images from this view
# Capture different camera viewpoint from different angles
# Set the initial seed on the object surface -- pending
# 3D spin images -- this is a graph problem
# 3D depth images -- this is only for depth estimation

import vtk
import numpy as np
import cv2
from vtk import *
import BTL_Registration
import open3d
import BTL_VIZ

class MultiCam():

    def __init__(self):

        # Brain stl -- this is important
        filename_brainstl = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_slice.stl'
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename_brainstl)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor_brainstl = vtk.vtkActor()
        actor_brainstl.SetMapper(mapper)
        self.brainstl_actor = actor_brainstl

        # Brain pcd
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()

        # Brain npy
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

    def SeedInit(self):

        # Initialize the whole seed together
        a = 1

        # Random sample the whole region together

    def TEST_VTK(self):

        # Set the axis actor
        transform = vtk.vtkTransform()
        transform.Translate(1.0, 0.0, 0.0)
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        actor_axis = axes

        # Read the whole project
        actor_brain = self.brainstl_actor

        # Read the camera object
        camera = vtk.vtkCamera()
        camera.SetPosition(0, 0, 50)
        camera.SetFocalPoint(0, 0, 0)

        # Create a renderer, render window, and interactor
        renderer = vtkRenderer()
        renderer.SetActiveCamera(camera)

        # Set RenderWindows
        renderWindow = vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add the actor to the scene
        renderer.AddActor(actor_brain)
        renderer.AddActor(actor_axis)
        renderer.SetBackground(1, 1, 1)  # Background color white

        # Render and interact
        renderWindow.Render()
        renderWindowInteractor.Start()

    def SpinImg(self):

        # spin images
        a = 1

    def SurfaceNormal(self):

        # Downsample the point cloud
        obj = self.brain_pcd

        # Downsample the pcd
        downpcd = open3d.voxel_down_sample(obj, voxel_size=1.0)

        # open3d.draw_geometries([downpcd])

        # Calculate the surface normal
        open3d.estimate_normals(downpcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 1, max_nn = 10))

        print(downpcd)

        # Show the final results
        open3d.draw_geometries([downpcd])

    def Downsample(self):

        # Downsample the point cloud
        obj = self.brain_pcd

        # Downsample the pcd
        downpcd = open3d.voxel_down_sample(obj, voxel_size = 1.0)

        # Show the final results
        open3d.draw_geometries([downpcd])

if __name__ == "__main__":

    test = MultiCam()
    test.SurfaceNormal()