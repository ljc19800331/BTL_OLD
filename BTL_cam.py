'''
ref1 plane object: https://lorensen.github.io/VTKExamples/site/Python/GeometricObjects/Plane/
'''

import vtk
import BTL_GL
import vtki
from vtki import examples
import numpy as np

def cam_test():

    # Create the color vector
    colors = vtk.vtkNamedColors()

    # Create a sphere
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0, 0, 10)
    sphereSource.SetRadius(1)
    sphereSource.SetPhiResolution(30)
    sphereSource.SetThetaResolution(30)
    sphereSource.Update()

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.6)
    actor.GetProperty().SetSpecularPower(30)
    actor.GetProperty().SetColor(colors.GetColor3d("LightSkyBlue"))

    # Set up the camera
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 30)
    camera.SetFocalPoint(0, 0, 10)
    camera.Azimuth(150)
    camera.Elevation(30)
    # camera.Dolly(1.5)

    # Set the axes
    transform = vtk.vtkTransform()
    transform.Translate(0.0, 0.0, 0.0)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.SetActiveCamera(camera)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    renderer.AddActor(axes)
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("MistyRose"))

    # Render and interact
    renderWindowInteractor.Initialize()
    renderWindow.Render()
    renderWindowInteractor.Start()

def cam_test_1():

    # The source file
    file_name = "/home/mgs/PycharmProjects/BTL_GS/BTL/vtki_test.vtk"

    # Read the source file.
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    scalar_range = output.GetScalarRange()

    # Create the mapper that corresponds the objects of the vtk file
    # into graphics elements
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(output)
    mapper.SetScalarRange(scalar_range)

    # Create the Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create the Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    # Create the RendererWindow
    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)

    # Create the RendererWindowInteractor and display the vtk_file
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()

def cam_plane():

    # Create a vtk object -- get the vtk data from vtki object
    a = 1
    test = BTL_GL.BTL_GL()
    img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain_mask_color.jpg'
    stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_500.stl'
    vtk_data = test.ColorTextureMap(img_path, stl_path)
    # print(type(vtk_data))

    # glyph = vtk.vtkGlyph3D()
    # glyph.SetInputData(vtk_data)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vtk_data)

    # Load the texture image

    # define the camera position

    # Capture the image and save to the folder

    # record the camra paramters

def cammodel():

    # Create an textured object
    test = BTL_GL.BTL_GL()
    test.img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/lena.jpg'
    test.stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_200.stl'
    test.colorstl_file = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/color_plane.npy'
    actor_texture, obj = test.ShowColorSTL()
    print("The obj is ", obj)

    # create a plane -- move the plane to the centroid
    colors = vtk.vtkNamedColors()
    # Set the background color
    colors.SetColor("BkgColor", [26, 51, 77, 255])
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetCenter(0.0, 0.0, 1)
    planeSource.SetNormal(0, 0, 1)
    planeSource.Update()
    plane = planeSource.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plane)
    actor_plane = vtk.vtkActor()
    actor_plane.SetMapper(mapper)
    actor_plane.GetProperty().SetColor(colors.GetColor3d("Cyan"))

    # Create a transform axes
    transform = vtk.vtkTransform()
    transform.Translate(0.0, 0.0, 0.0)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    # Create a multiple camera model
    camera = vtk.vtkCamera()
    camera.SetPosition(50, 50, 150)
    camera.SetFocalPoint(50, 50, 1)
    camera.Azimuth(30)
    camera.Elevation(30)
    renderer = vtk.vtkRenderer()
    renderer.SetActiveCamera(camera)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    renderer.AddActor(axes)
    renderer.AddActor(actor_plane)
    renderer.AddActor(actor_texture)

    # Render and interact
    renderWindowInteractor.Initialize()
    renderWindow.Render()
    renderWindowInteractor.Start()

def vtki_test():

    # Load and read the data
    img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/lena.jpg'
    stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_200.stl'

    # Create the camera object -- this is im
    plotter = vtki.Plotter()
    texture = vtki.load_texture(img_path)
    obj = vtki.read(stl_path)
    obj.texture_map_to_plane( inplace = True )
    _ = plotter.add_mesh(obj, texture = texture)
    plotter.camera_position = [(50.0, 50.0, 50), (50.0, 50.0, 1.0), (0.0, 1.0, 0.0)]
    cam = plotter.camera_position
    # plotter.screenshot('test.png')
    plotter.show()

    # Create the camera objects -- this is im
    camera = vtk.vtkCamera()
    camera.SetPosition(50, 50, 150) # We define the x as the radius to the spherical model -- this is im
    camera.SetFocalPoint(50, 50, 0)
    camera.Azimuth(30)
    camera.Elevation(30)
    camera.Roll(0)
    print("The camera parameter is ", camera)
    print("The position of the camera is ", camera.GetPosition())

def actor_camera(p_center, p_direction, p_scale):

    camCS = vtk.vtkConeSource()
    camCS.SetDirection(p_direction)
    camCS.SetHeight(10)
    camCS.SetResolution(12)
    camCS.SetRadius(5)
    camCS.SetCenter(p_center)
    camAPD = vtk.vtkAppendPolyData()
    camAPD.AddInputConnection(camCS.GetOutputPort())
    camMapper = vtk.vtkPolyDataMapper()
    camMapper.SetInputConnection(camAPD.GetOutputPort())
    camActor = vtk.vtkLODActor()
    camActor.SetMapper(camMapper)
    camActor.SetScale(p_scale)

    return camActor

def CamModel():

    # Define the color background
    colors = vtk.vtkNamedColors()
    colors.SetColor("SpikeColor", [255, 77, 255, 255])

    # Create a rendering window, renderer and interactor.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Define the coordinate axes
    transform = vtk.vtkTransform()
    transform.Translate(0.0, 0.0, 0.0)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    # Define the texture object
    img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain_texture.jpg'
    stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_200.stl'
    plotter = vtki.Plotter()
    texture = vtki.load_texture(img_path)
    obj = vtki.read(stl_path)
    obj.texture_map_to_plane( inplace = True )
    _ = plotter.add_mesh(obj, texture = texture )

    # Define the camera angles in a spherical model
    x_mesh = np.linspace(-45, 45, 10)
    y_mesh = np.linspace(-45, 45, 10)

    focal_center = (35.0, 30.0, 1.0)
    pos_center = (35.0, 30.0, 50)
    # focal_coordinate = (50.0,  50.0, 1.0)
    focal_coordinate = focal_center

    for i in range(9):
        for j in range(9):

            # Get the camera coordinate in the spherical model -- this is important
            angle_Azimuth = x_mesh[i]
            angle_Elevation = y_mesh[j]
            camera = vtk.vtkCamera()
            camera.SetPosition(pos_center)         # We define the x as the radius to the spherical model -- this is im
            camera.SetFocalPoint(focal_center)
            camera.Azimuth(angle_Azimuth)           # 0 to 45 with 5 as uniform sample
            camera.Elevation(angle_Elevation)       # 0 to 45 with 5 as uniform sample
            camera.Roll(0)                          # no rolling in consideration -- this is im
            cam_coordinate = camera.GetPosition()
            print("The camera coordinate is ", cam_coordinate)

            plotter.camera_position = [cam_coordinate, focal_coordinate, (0.0, 1.0, 0.0)]
            cam = plotter.camera_position

            # Define the camera model -- this is important
            p_center = cam_coordinate
            p_direction = np.asarray(cam_coordinate) - np.asarray(focal_coordinate)
            p_scale = (1, 1, 1)
            camActor = actor_camera(p_center, p_direction, p_scale)
            # plotter.add_actor(axes)
            # plotter.add_actor(camActor)
            img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Screenshot/' + str(angle_Azimuth) + '_' + str(angle_Elevation) + '.png'
            plotter.show_axes()
            plotter.screenshot(img_path)
            # plotter.show(screenshot = img_path)

    plotter.show(screenshot='test.png')

# vtki_test()
CamModel()