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

class VtkPointCloud:

    # The depth color mapping case
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        # Initializing for the problem
        self.maxNumPoints = maxNumPoints            # max number points
        self.vtkPolyData = vtk.vtkPolyData()        # poly data obj
        self.clearPoints()                          # modify depth data
        mapper = vtk.vtkPolyDataMapper()            # Design the mapper info
        mapper.SetInputData(self.vtkPolyData)       # Scalar problem -- color depth
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):

        # The main reason why this includes the depth data
        # The main output and update data is shown in the following:
        # 1. vtkCells.Modified(), 2.vtkPoints.Modified(), 3.vtkDepth.Modified()

        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        # Update the results
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

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

class sec_viz():

    def __init__(self):
        self.a = 1
        self.txtbrain_x = '/home/maguangshen/PycharmProjects/pcltest/Data/brain_x.txt'
        self.txtbrain_y = '/home/maguangshen/PycharmProjects/pcltest/Data/brain_y.txt'
        self.txtbrain_z = '/home/maguangshen/PycharmProjects/pcltest/Data/brain_z.txt'

    def PointUpdate(self, npy_data, x_pos, y_pos):

        # This is used to update the point sets for each position
        # Input: npy_brain, index_initial, x_pos(the laser spot on x), y_pos(laser spot on y)
        # Output: The index of the point
        # Update each step for the point sets

        # Set the point region
        range = 0.025
        x_range = [x_pos - range, x_pos + range]
        y_range = [y_pos - range, y_pos + range]

        # The region for x
        x_idx = np.transpose(np.array(np.where(npy_data[:, 0] > x_range[0])))
        x_obj = np.reshape(npy_data[x_idx, :], (len(x_idx), 3))
        x_idx = np.transpose(np.array(np.where(x_obj[:, 0] < x_range[1])))
        x_obj = np.reshape(x_obj[x_idx, :], (len(x_idx), 3))
        # x_test = SEC_DataProcess.npy2vtk(x_obj[:, 0], x_obj[:, 1], x_obj[:, 2])

        # The region for y
        y_idx = np.transpose(np.array(np.where(x_obj[:, 1] > y_range[0])))
        y_obj = np.reshape(x_obj[y_idx, :], (len(y_idx), 3))
        y_idx = np.transpose(np.array(np.where(y_obj[:, 1] < y_range[1])))
        y_obj = np.reshape(y_obj[y_idx, :], (len(y_idx), 3))

        # Remove the value > mean(z)
        # print(y_obj)
        z_mean = np.mean(y_obj[:,2])
        print(z_mean)
        spot_idx = np.transpose(np.array(np.where(y_obj[:, 2] > z_mean)))
        npy_spot = np.reshape(y_obj[spot_idx, :], (len(spot_idx), 3))
        vtk_spot = SEC_PRE.npy2vtk(npy_spot)
        # y_test = SEC_DataProcess.npy2vtk(y_obj[:, 0], y_obj[:, 1], y_obj[:, 2])

        return vtk_spot, npy_spot

    def SceneViz(self, vtk_brain, vtk_spot, npy_brain, npy_spot):

        # input the vtk data -- the list of th npy data
        # Show all the vtk data at the same time

        # Axis
        transform = vtk.vtkTransform()  # transformation of a 3D axis
        transform.Translate(0.0, 0.0, 0.0)  # Remain the default setting
        axes = vtk.vtkAxesActor()  # Add the axis actor
        axes.SetUserTransform(transform)

        # Read the actors
        actor_brain = ActorColor(npy_brain, vec_color=[255, 0, 0])
        actor_spot = ActorColor(npy_spot, vec_color=[255, 255, 255])

        # Design the renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor_brain)
        renderer.AddActor(actor_spot)
        renderer.AddActor(axes)
        renderer.SetBackground(0.2, 0.3, 0.4)  # set background color
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # renderWindowInteractor.SetRenderWindow(renderWindow)

        # Begin Interaction
        renderWindow.Render()
        # renderWindowInteractor.Start()

        # remove the actors
        renderer.RemoveActor(actor_spot)
        renderWindow.Render()

        return renderWindow, renderer

    def ScanPath(self):

        # Input: Laser spot / starting point
        # Output: Laser path in the local coordinate
        # The rule is that the starting point is always on the left
        x_o = 1.75; y_o = 1.75    # The origin of the scanning point
        unit = 0.05    # inch
        x_range = 0.5   # scan size
        y_range = 0.5   # scan size
        Nx = x_range / unit; Ny = y_range / unit

        # mesh grid for the path
        x_mesh = np.linspace(x_o, x_o + x_range, Nx)
        y_mesh = np.linspace(y_o, y_o + y_range, Ny)
        xv, yv = np.meshgrid(x_mesh, y_mesh)
        xv = np.reshape(xv, (int(xv.shape[0]) * int(xv.shape[1]))  )
        yv = np.reshape(yv, (int(yv.shape[0]) * int(yv.shape[1]))  )
        # print(int(xv.shape[0]))
        # print(np.reshape(xv, (int(xv.shape[0]) * int(xv.shape[1]))  ))
        return xv, yv

    def ShootPath(self):
        # The shoot path -- user can shoot whatever point to shoot
        # Manual control by the surgeon
        a = 1

    def test_1(self):

        # ACTOR: Number of actors
        # Update the new point cloud
        # Update each step on the surface point cloud

        # load the brain model and viz
        map = SEC_MAP.sec_map()
        txt = [self.txtbrain_x, self.txtbrain_y, self.txtbrain_z]
        theta = [180, 180, 0]
        vtk_brain, npy_brain, x_brain, y_brain, z_brain = map.Brain_Init(txt[0], txt[1], txt[2], theta)

        # Show the initial position -- The brain model
        theta = [0, 0, 0]
        R_tform = SEC_MAP.AglTransform(theta[0], theta[1], theta[2])
        pc_tform = np.matmul(npy_brain, R_tform)
        pc_tform = np.asarray(pc_tform)                         # The difference between ndarray and asanyarray
        pc_tform = SEC_MAP.Npy2Origin(pc_tform)

        # Show the initial position -- The point index
        x_pos = 1.0; y_pos = 1.0
        vtk_spot, npy_spot = self.PointUpdate(pc_tform, x_pos, y_pos)

        # Show the scene -- First position
        renderWindow, renderer = self.SceneViz(vtk_brain, vtk_spot, npy_brain, npy_spot)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)

        # Update the scene
        # Design the axes transform
        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)

        # Design the active camera model
        camera = vtk.vtkCamera()
        camera.SetPosition(1, 1, 1)
        camera.SetFocalPoint(0, 0, 0)

        camera_1 = vtk.vtkCamera()
        camera_1.SetPosition(1, 1, 1)
        camera_1.SetFocalPoint(0, 0, 0)

        # Design the actors
        # actor_brain = ActorColor(npy_brain, vec_color=[255, 0, 0])
        actor_stl = stl_viz()

        # Design the renderer
        ren_1 = vtk.vtkRenderer()
        ren_1.SetActiveCamera(camera)
        renWin_1 = vtk.vtkRenderWindow()
        renWin_1.AddRenderer(ren_1)
        renWin_1.SetSize(400, 400)
        renWin_1.SetWindowName("Right")

        ren_2 = vtk.vtkRenderer()
        ren_2.SetActiveCamera(camera)
        renWin_2 = vtk.vtkRenderWindow()
        renWin_2.AddRenderer(ren_2)
        renWin_2.SetSize(400, 400)
        renWin_2.SetWindowName("Left")

        # Add the actors and renders
        # Add the axes with the actor
        # iren = vtk.vtkRenderWindowInteractor()
        # iren.SetRenderWindow(renWin_1)
        ren_1.AddActor(axes)
        # ren.AddActor(actor_brain)
        ren_1.AddActor(actor_stl)

        ren_2.AddActor(axes)
        ren_2.AddActor(actor_stl)

        campos = np.zeros(3);   # The position of the camera focal point
        delta = 0; count = 0

        # read the scanning path
        xv, yv = self.ScanPath()

        while (True):
            count += 1
            print(count)
            delta += 0.05
            # x_center = x_pos + delta
            # y_center = y_pos + delta
            x_center = xv[count]
            y_center = yv[count]
            vtk_spot, npy_spot = self.PointUpdate(npy_brain, x_center, y_center)
            actor_spot = ActorColor(npy_spot, vec_color=[255, 0, 0])

            # convert the points to the visible spot
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(np.mean (npy_spot[:,0]), np.mean (npy_spot[:,1]), np.mean (npy_spot[:,2]))
            sphereSource.SetRadius(0.05)
            sphereSource.Update()
            mapper = vtk.vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor_ball = vtk.vtkActor()
            actor_ball.SetMapper(mapper)
            actor_ball.GetProperty().SetColor(1, 0, 0)  # (R,G,B)

            # update the actor and camera
            campos[0] = np.mean (npy_spot[:,0]); campos[1] = np.mean (npy_spot[:,1]); campos[2] = np.mean (npy_spot[:,2])
            camera.SetPosition(campos[0], campos[1], campos[2] * 3)
            camera.SetFocalPoint(campos[0], campos[1], campos[2])

            camera_1.SetPosition(campos[0], campos[1], campos[2] * 10)
            camera_1.SetFocalPoint(campos[0], campos[1], campos[2])

            ren_1.SetActiveCamera(camera)
            ren_1.AddActor(actor_spot)
            ren_1.AddActor(actor_ball)
            renWin_1.Render()
            time.sleep(0.1)
            # ren_1.RemoveActor(actor_spot)
            # ren_1.RemoveActor(actor_ball)
            # iren.Initialize()
            renWin_1.Render()
            #
            ren_2.SetActiveCamera(camera_1)
            ren_2.AddActor(actor_spot)
            ren_2.AddActor(actor_ball)
            renWin_2.Render()
            time.sleep(0.1)
            # ren_2.RemoveActor(actor_spot)
            # ren_2.RemoveActor(actor_ball)
            # iren.Initialize()
            renWin_2.Render()

def DoublenWin():
    # create two cameras
    cameraR = vtk.vtkCamera()
    cameraR.SetPosition(0, 0, 200)
    cameraR.SetFocalPoint(0, 0, 0)

    cameraL = vtk.vtkCamera()
    cameraL.SetPosition(40, 0, 200)
    cameraL.SetFocalPoint(0, 0, 0)

    # create a rendering window and renderer
    renR = vtk.vtkRenderer()
    renR.SetActiveCamera(cameraR)

    renL = vtk.vtkRenderer()
    renL.SetActiveCamera(cameraL)

    # create source
    viz = sec_viz()
    map = SEC_MAP.sec_map()
    txt = [viz.txtbrain_x, viz.txtbrain_y, viz.txtbrain_z]
    theta = [180, 180, 0]
    vtk_brain, npy_brain, x_brain, y_brain, z_brain = map.Brain_Init(txt[0], txt[1], txt[2], theta)

    # create render window
    renWinR = vtk.vtkRenderWindow()
    renWinR.AddRenderer(renR)
    renWinR.SetWindowName("Right")

    renWinL = vtk.vtkRenderWindow()
    renWinL.AddRenderer(renL)
    renWinL.SetWindowName("Left")

    # create a render window interactor
    irenR = vtk.vtkRenderWindowInteractor()
    irenR.SetRenderWindow(renWinR)

    irenL = vtk.vtkRenderWindowInteractor()
    irenL.SetRenderWindow(renWinL)

    actor = vtk_brain.vtkActor

    # assign actor to the renderer
    renR.AddActor(actor)
    renL.AddActor(actor)

    # enable user interface interactor
    renWinR.Render()
    renWinL.Render()
    irenR.Initialize()
    irenL.Initialize()

    # Create callback object for camera manipulation
    cb = vtkGyroCallback()
    cb.rightCam = renR.GetActiveCamera()  # cameraR
    cb.leftCam = renL.GetActiveCamera()  # cameraL
    cb.irenR = irenR
    cb.irenL = irenL

    irenR.ID = 1
    irenL.ID = 2
    irenR.AddObserver('TimerEvent', cb.execute)
    irenL.AddObserver('TimerEvent', cb.execute)
    timerIDR = irenR.CreateRepeatingTimer(100)
    timerIDL = irenL.CreateRepeatingTimer(100)

    irenL.Start()
    irenR.Start()

def vtk_pc(vtk_list):

    # vtk_list = [vtk_data1, vtk_data2, vtk_data3 ...]
    # visualize the single pc data
    # determine a 3D axis

    transform = vtk.vtkTransform()          # transformation of a 3D axis
    transform.Translate(0.0, 0.0, 0.0)      # Remain the default setting
    axes = vtk.vtkAxesActor()               # Add the axis actor
    axes.SetUserTransform(transform)

    # Renderer -- with a loop
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.2, .3, .4)      # set background color
    renderer.ResetCamera()
    renderer.AddActor(axes)
    for item in vtk_list:
        renderer.AddActor(item.vtkActor)

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

def stl_viz():

    # Input: stlname
    # Output: actor for the stl
    filename = '/home/maguangshen/PycharmProjects/pcltest/Data/brain_tumor_scaled.stl'
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

def stl_change():

    # This function is used to change the parameters of stl file
    your_mesh = mesh.Mesh.from_file('/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_slice.stl')
    your_mesh.v0, your_mesh.v1, your_mesh.v2

def CamView():

    # This function is used to test the Camera view
    # Create a sphere
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0.0, 0.0, 0.0)
    sphereSource.SetRadius(10)
    sphereSource.Update()

    # Create a mapper and actor
    mapper = vtk.vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    camera = vtk.vtkCamera()
    camera.SetPosition(100, 100, 100)
    camera.SetFocalPoint(10, 0, 0)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.SetActiveCamera(camera)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Background color white

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

def ActorColor(npy_data, vec_color):

    # from npy to array
    x = npy_data[:,0]
    y = npy_data[:,1]
    z = npy_data[:,2]

    # Set up the point and vertices
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()

    # Set up the color objects
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")

    length = int(len(x))

    # Set up the point and vertice object
    for i in range(length):
        p_x = x[i]
        p_y = y[i]
        p_z = z[i]
        id = Points.InsertNextPoint(p_x, p_y, p_z)                                             # The point coordinate is here
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(id)
        Colors.InsertNextTuple3(vec_color[0], vec_color[1], vec_color[2])                      # set as red color for this purpose

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    polydata.SetVerts(Vertices)
    polydata.GetPointData().SetScalars(Colors)  # Set the color points for the problem
    polydata.Modified()

    # Set up the actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def ParseSIG(signum, stack):
    print signum
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