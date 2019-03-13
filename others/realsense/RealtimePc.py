# Reference: https://github.com/toinsson/pyrealsense/blob/master/examples/show_vtk.py
# https://stackoverflow.com/questions/44188762/update-live-pointcloud-data-in-vtk-python
import time
import threading
import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
import sys
sys.path.append('/usr/local/lib')
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import BTL_VIZ
from BTL_VIZ import *
import TextureMap
from TextureMap import *
import BTL_DataConvert
from BTL_DataConvert import *

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)
curr_frame = 0

# the point cloud obejct
pc = rs.pointcloud()

# Design the Actor objects for all the vtk files
class VTKActorWrapper(object):

    def __init__(self, nparray):

        super(VTKActorWrapper, self).__init__()
        self.nparray = nparray
        nCoords = nparray.shape[0]
        nElem = nparray.shape[1]

        self.verts = vtk.vtkPoints()    # vtkPoints object
        self.cells = vtk.vtkCellArray() # cellarry
        self.scalars = None

        self.pd = vtk.vtkPolyData()
        self.verts.SetData(vtk_np.numpy_to_vtk(nparray))
        self.cells_npy = np.vstack([np.ones(nCoords,dtype=np.int64),
                               np.arange(nCoords,dtype=np.int64)]).T.flatten()
        self.cells.SetCells(nCoords,vtk_np.numpy_to_vtkIdTypeArray(self.cells_npy))
        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputDataObject(self.pd)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToPoints()
        self.actor.GetProperty().SetColor(0.0,1.0,0.0)

    def update(self, threadLock, update_on):
        thread = threading.Thread(target=self.update_actor, args=(threadLock, update_on))
        thread.start()

    def update_actor(self, threadLock, update_on):

        while (update_on.is_set()):
            time.sleep(0.01)
            threadLock.acquire()

            # update the point sets
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            points = pc.calculate(depth)
            vtx = np.asanyarray(points.get_vertices())
            npy_data = Bracket2npy(vtx)

            self.nparray[:] = npy_data # rs.points.reshape(-1,3)  # points have to be reshape
            self.pd.Modified()
            threadLock.release()

class VTKVisualisation(object):

    def __init__(self, threadLock, actorWrapper, axis=True,):

        super(VTKVisualisation, self).__init__()

        self.threadLock = threadLock
        self.ren = vtk.vtkRenderer()
        self.ren.AddActor(actorWrapper.actor)

        self.axesActor = vtk.vtkAxesActor()
        self.axesActor.AxisLabelsOff()
        self.axesActor.SetTotalLength(1, 1, 1)
        self.ren.AddActor(self.axesActor)

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)

        ## IREN
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.Initialize()

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)

        self.iren.AddObserver("TimerEvent", self.update_visualisation)
        dt = 30 # ms
        timer_id = self.iren.CreateRepeatingTimer(dt)

    def update_visualisation(self, obj=None, event=None):
        time.sleep(0.01)
        self.threadLock.acquire()
        self.ren.GetRenderWindow().Render()
        self.threadLock.release()

def main():

    # Set up the theading object and function
    update_on = threading.Event()
    update_on.set()
    threadLock = threading.Lock()

    # get the point cloud numpy array
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    points = pc.calculate(depth)
    vtx = np.asanyarray(points.get_vertices())
    npy_data = Bracket2npy(vtx)

    # Map the point object to the wrapper
    actorWrapper = VTKActorWrapper(npy_data)  # notice pc must to be nparray
    actorWrapper.update(threadLock, update_on)

    viz = VTKVisualisation(threadLock, actorWrapper)
    viz.iren.Start()
    update_on.clear()

main()
pipeline.stop()