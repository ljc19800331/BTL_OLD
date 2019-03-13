# Textured mapping with Python OpenGL
# This is the example code
# ! /usr/bin/env python
'''
ref: https://gist.github.com/Jerdak/7364746
ref: https://www.programcreek.com/python/example/12565/vtk.vtkSphereSource
ref: https://github.com/Kitware/VTK/tree/master/Examples/Modelling/Python
ref: https://lorensen.github.io/VTKExamples/site/Python/DataManipulation/LineOnMesh/
ref: http://vtk.1045678.n5.nabble.com/How-to-color-a-sphere-source-in-an-appendpolydata-object-td5736260.html
ref: https://github.com/Kitware/VTK/blob/master/Examples/VisualizationAlgorithms/Python/GenerateTextureCoords.py
ref: https://gist.github.com/somada141/acefac8a6360cba21f3a
ref: https://public.kitware.com/pipermail/vtkusers/2015-July/091487.html
ref: https://www.cnblogs.com/21207-iHome/p/6597949.html

Method 1: Triangle and mesh
https://vtk.org/Wiki/VTK/Examples/Python/PolyData/SubdivisionFilters

Method 2: Sphere mapping

Simple example demonstrating how to take a screen-shot
Possible solution for the video frame:
    1. Render with close sphere ball -- the render problem waiting for the answer
    2. 3D reconstruction -- from sparse point cloud -- pending from MATLAB
    3. Triangulation mesh operation -- triangulation ID for each point cloud for the surface -- pending
    4. Texutre mapping to a sphere
    5. How to smooth the point cloud -- this is the main problem
    6. Possible solution includes mesh smoothing -- maybe in meshlab
'''
import vtk
from vtk import *
import numpy as np
import BTL_GL
import os
import string
from vtk.util.misc import vtkGetDataRoot
import scipy.io as sio
import cv2

def Test_1():

    VTK_DATA_ROOT = vtkGetDataRoot()

    # Load the data for point cloud and colorized vectors
    nc = vtk.vtkNamedColors()
    test = BTL_GL.BTL_GL()
    pcd, pc, pc_color, MeshGrid, img_color = test.GL_NonFlat()


    # The sphere model
    sphere = vtk.vtkSphereSource()
    sphere.SetPhiResolution(5)
    sphere.SetThetaResolution(5)
    sphere.SetRadius(0.0025 * 2)

    # The sphere object is vtkSphereSource
    print("The sphere object is ", sphere)

    # Add the points to object
    Points = vtk.vtkPoints()

    # Add the color array to the object
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    Colors.InsertNextTuple3(255, 0, 0)

    # vtkUnsignedCharArray --
    print("The color object is ", Colors)

    for i in range(len(pc)):
        Points.InsertNextPoint(pc[i, 0], pc[i, 1], pc[i, 2])
        Colors.InsertNextTuple3(pc_color[i, 0] / 255, pc_color[i, 1] / 255, pc_color[i, 2] / 255)

    # Design the polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    # Check the sphere object
    sphere.GetOutput().GetCellData().SetScalars(Colors)
    sphere.Update()

    appendData = vtk.vtkAppendPolyData()
    appendData.AddInputConnection(sphere.GetOutputPort())
    appendData.Update()

    # Design the mapper
    point_mapper = vtk.vtkGlyph3DMapper()
    point_mapper.SetInputData(polydata)
    point_mapper.SetSourceConnection(appendData.GetOutputPort())
    point_mapper.SetScalarModeToUseCellData()

    # Set background color
    actor = vtk.vtkActor()
    actor.SetMapper(point_mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().BackfaceCullingOn()

    ren = vtk.vtkRenderer()
    ren.SetBackground(.2, .3, .4)
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renWin)

    # Begin Interaction
    renWin.Render()
    renderWindowInteractor.Start()

def Test_2():

    # Generate a meshpoints
    test = BTL_GL.BTL_GL()
    pcd, pc, pc_color, MeshGrid, img_color = test.GL_NonFlat()

    # print("The mesh grid is ", MeshGrid)
    x_mesh = MeshGrid[0]
    y_mesh = MeshGrid[1]
    z_mesh = MeshGrid[2]

    [size_h, size_w] = z_mesh.shape
    topography = np.zeros([size_h, size_w])

    # The color image
    print("The matrix of the color image is ", img_color.shape)

    for i in range(size_h):
        for j in range(size_w):
            # print(z_mesh[i,j])
            topography[i][j] = z_mesh[i,j]

    # Generate the mesh and data from
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    # Build the mesh manually
    count = 0
    for i in range(size_h - 1):
        for j in range(size_w - 1):

            z1 = topography[i][j]
            z2 = topography[i][j + 1]
            z3 = topography[i + 1][j]

            # print("The currrent point coordinate is ", [i, j, z1])
            points.InsertNextPoint(x_mesh[i,j], y_mesh[i,j], z1)
            points.InsertNextPoint(x_mesh[i,j+1], y_mesh[i, j + 1], z2)
            points.InsertNextPoint(x_mesh[i + 1, j], y_mesh[i+1, j], z3)

            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, count)
            triangle.GetPointIds().SetId(1, count + 1)
            triangle.GetPointIds().SetId(2, count + 2)
            triangles.InsertNextCell(triangle)

            # Assign the colors
            colors.InsertNextTuple3(int(img_color[i, j, 0]), int(img_color[i, j, 1]), int(img_color[i, j, 2]))
            colors.InsertNextTuple3(int(img_color[i, j+1, 0]), int(img_color[i, j+1, 1]), int(img_color[i, j+1, 2]))
            colors.InsertNextTuple3(int(img_color[i+1, j, 0]), int(img_color[i+1, j, 1]), int(img_color[i+1, j, 2]))
            count += 3

    # Create a polydata object
    trianglePolyData = vtk.vtkPolyData()

    # Add the geometry and topology to the polydata
    trianglePolyData.SetPoints(points)
    trianglePolyData.GetPointData().SetScalars(colors)
    trianglePolyData.SetPolys(triangles)

    # Create a mapper and actor for smoothed dataset
    point_mapper = vtk.vtkPolyDataMapper()
    point_mapper.SetInputData(trianglePolyData)
    actor = vtk.vtkActor()
    actor.SetMapper(point_mapper)

    actor = vtk.vtkActor()
    actor.SetMapper(point_mapper)

    # Visualize
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add actors and render
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Background color white
    renderWindow.SetSize(600, 600)
    renderWindow.Render()
    renderWindowInteractor.Start()

def Test_3():

    # Map 3D colorized point cloud with spheres
    a = 1

def Test_4():

    # Load the data object
    '''
    pcd: object of pcd
    pc: point coordinates, M x 3
    pc_color: color vectors, M x 3
    img_color: color image for textured mapping
    '''

    test = BTL_GL.BTL_GL()
    pcd, pc, pc_color, MeshGrid, img_color = test.GL_NonFlat()

    # pc_color = np.load('/home/mgs/PycharmProjects/BTL_GS/BTL/Color_stl.npy')

    # Define the points polydata
    points = vtk.vtkPoints()
    for i in range(len(pc)):
        points.InsertNextPoint(pc[i, 0], pc[i, 1], pc[i, 2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Define the color to the polydata
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(polydata.GetNumberOfPoints())
    for i in range(len(pc_color)):
        colors.InsertTuple3(i, pc_color[i, 0], pc_color[i, 1], pc_color[i, 2])

    # Connect the point object to the color object
    polydata.GetPointData().SetScalars(colors)

    # Define the VertexGlyphFilter
    vgf = vtk.vtkVertexGlyphFilter()
    vgf.SetInputData(polydata)
    vgf.Update()
    pcd = vgf.GetOutput()

    # Define the mapper
    point_mapper = vtk.vtkPolyDataMapper()
    point_mapper.SetInputData(pcd)

    # Define the actor
    actor = vtk.vtkActor()
    actor.SetMapper(point_mapper)
    actor.GetProperty().SetPointSize(5)
    actor.GetProperty().RenderPointsAsSpheresOn()

    ren = vtk.vtkRenderer()
    ren.SetBackground(.2, .3, .4)
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renWin)

    # Begin Interaction
    renWin.Render()
    renderWindowInteractor.Start()

def Test_5():

    # Load the color and greyscale image
    test = BTL_GL.BTL_GL()
    pcd, pc, pc_color, MeshGrid, img_color = test.GL_NonFlat()
    img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    [h, w] = img_grey.shape
    img_grey_use = np.reshape(img_grey, (h * w, 1))

    # Load the stl object
    filename = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/test.stl'
    f = vtk.vtkSTLReader()
    f.SetFileName(filename)
    f.Update()  # This is necessary to have the data ready to read.

    obj = f.GetOutputDataObject(0)
    min_z, max_z = obj.GetBounds()[4:]

    # Assign the loopup table
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(min_z, max_z)
    lut.Build()

    # heights = vtk.vtkDoubleArray()
    # heights.SetName("Z_Value")

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")

    # Assign the color vector
    for i in range(obj.GetNumberOfPoints()):
        # z = pc_color[i, :]
        # Colors.InsertNextTuple3(pc_color[i, 0], pc_color[i, 1], pc_color[i, 2])
        z = obj.GetPoint(i)[-1]
        # heights.InsertNextValue3(z)

    print("The height is ", Colors)

    obj.GetPointData().SetScalars(Colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputDataObject(obj)
    mapper.SetScalarRange(min_z, max_z)
    mapper.SetLookupTable(lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(.1, .2, .4)

    renw = vtk.vtkRenderWindow()
    renw.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renw)

    renw.Render()
    iren.Start()

if __name__ == "__main__":

    # Test_1()
    # Test_2()
    Test_4()
    # Test_5()