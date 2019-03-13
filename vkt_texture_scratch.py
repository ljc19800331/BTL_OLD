'''
Texture mapping from scratch
Method 1: build from scratch from vtki
Method 2: build from vtki data object -- not recommended
'''
import vtk

colors = vtk.vtkNamedColors()

# Set the colors.
colors.SetColor("AzimuthArrowColor", [255, 77, 77, 255])
colors.SetColor("ElevationArrowColor", [77, 255, 77, 255])
colors.SetColor("RollArrowColor", [255, 255, 77, 255])
colors.SetColor("SpikeColor", [255, 77, 255, 255])
colors.SetColor("UpSpikeColor", [77, 255, 255, 255])

# Create a rendering window, renderer and interactor.
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Create a camera model.
camCS = vtk.vtkConeSource()
camCS.SetHeight(1.5)
camCS.SetResolution(12)
camCS.SetRadius(0.4)

camCBS = vtk.vtkCubeSource()
camCBS.SetXLength(1.5)
camCBS.SetZLength(0.8)
camCBS.SetCenter(0.4, 0, 0)

camAPD = vtk.vtkAppendPolyData()
camAPD.AddInputConnection(camCBS.GetOutputPort())
camAPD.AddInputConnection(camCS.GetOutputPort())

camMapper = vtk.vtkPolyDataMapper()
camMapper.SetInputConnection(camAPD.GetOutputPort())
camActor = vtk.vtkLODActor()
camActor.SetMapper(camMapper)
camActor.SetScale(2, 2, 2)

# Add the actors to the renderer, set the background and size.
ren.AddActor(camActor)
ren.SetBackground(colors.GetColor3d("SlateGray"))

cam1 = (ren.GetActiveCamera())
ren.ResetCamera()
cam1.Azimuth(150)
cam1.Elevation(30)
cam1.Dolly(1.5)
ren.ResetCameraClippingRange()

iren.Initialize()
iren.Start()




# colors = vtk.vtkNamedColors()
#
# # Set the background color.
# colors.SetColor("BkgColor", [26, 51, 77, 255])
#
# # Create a plane
# planeSource = vtk.vtkPlaneSource()
# planeSource.SetCenter(1.0, 0.0, 0.0)
# planeSource.SetNormal(1.0, 0.0, 1.0)
# planeSource.Update()
#
# plane = planeSource.GetOutput()
#
# # Create a mapper and actor
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputData(plane)
#
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)
# actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))
#
# # Create a renderer, render window and interactor
# renderer = vtk.vtkRenderer()
# renderWindow = vtk.vtkRenderWindow()
# renderWindow.SetWindowName("Plane")
# renderWindow.AddRenderer(renderer)
# renderWindowInteractor = vtk.vtkRenderWindowInteractor()
# renderWindowInteractor.SetRenderWindow(renderWindow)
#
# # Add the actors to the scene
# renderer.AddActor(actor)
# renderer.SetBackground(colors.GetColor3d("BkgColor"))
#
# # Render and interact
# renderWindow.Render()
# renderWindowInteractor.Start()