# This function is only for the pcl test
# Goals:
# 1. test different function and show the viz result
# 2. ICP and other functions
# 3. Segmentation
# 4. SAC -- point cloud stiching method
# 5. Smoothing method
# 6. Filtering
# 7. Registration

from __future__ import print_function
import numpy as np
from numpy import cos, sin
import unittest     # What is the unittest function here?
import pcl
import vtk
import SEC_VIZ
import time

class pcltest():
    def __init__(self):
        # the brain txt files
        test = 1

    def exp_icp(self,):
        a = 1

class TestICP():
    def __init__(self):
        test = 2

    def setUp(self):

        # Set up the transform
        theta = [-1, 0.5, 1]
        rot_x = [[1, 0, 0],
                 [0, cos(theta[0]), -sin(theta[0])],
                 [0, sin(theta[0]), cos(theta[0])]]
        rot_y = [[cos(theta[1]), 0, sin(theta[1])],
                 [0, 1, 0],
                 [-sin(theta[1]), 0, cos(theta[1])]]
        rot_z = [[cos(theta[2]), -sin(theta[1]), 0],
                 [sin(theta[2]), cos(theta[1]), 0],
                 [0, 0, 1]]
        transform = np.dot(rot_x, np.dot(rot_y, rot_z))

        # source
        pcd_name = "/home/maguangshen/PycharmProjects/BTL_GS/Data/test.pcd"
        source = np.asarray(pcl.load(pcd_name))
        self.pcd_source = source
        self.source = pcl.PointCloud(source.astype(np.float32))
        self.pcd_update = source
        self.update = pcl.PointCloud(self.pcd_update .astype(np.float32))

        # Target
        target =  np.dot(source, transform)
        self.pcd_target = target
        self.target = pcl.PointCloud(target.astype(np.float32))

    def pcd2vtk(self):

        x_s = np.asarray(self.pcd_source[:,0])
        y_s = np.asarray(self.pcd_source[:,1])
        z_s = np.asarray(self.pcd_source[:,2])

        x_t = np.asarray(self.pcd_target[:,0])
        y_t = np.asarray(self.pcd_target[:,1])
        z_t = np.asarray(self.pcd_target[:,2])

        vtk_source = SEC_VIZ.VtkPointCloud()
        vtk_target = SEC_VIZ.VtkPointCloud()

        for k in range(len(x_s)):
            point_s = ([x_s[k], y_s[k], z_s[k]])
            point_t = ([x_t[k], y_t[k], z_t[k]])
            vtk_source.addPoint(point_s)
            vtk_target.addPoint(point_t)
        self.vtk_source = vtk_source
        self.vtk_target = vtk_target

        return vtk_source, vtk_target

    def testICP(self):
        icp = self.source.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(self.source, self.target, max_iter=1000)
        transf_s2t = np.transpose(transf)[0:3,0:3]
        print(transf_s2t)

        # transform the model
        Res = np.dot(self.pcd_source, transf_s2t)
        print(Res)
        vtk_res = SEC_VIZ.VtkPointCloud()
        x_res = np.asarray(Res[:, 0])
        y_res = np.asarray(Res[:, 1])
        z_res = np.asarray(Res[:, 2])
        for k in range(len(x_res)):
            point_res = ([x_res[k], y_res[k], z_res[k]])
            vtk_res.addPoint(point_res)
        SEC_VIZ.vtk_pc([vtk_res, vtk_target])
        return converged, transf, estimate, fitness

    def ICPupdate(self):
        # Return the transform
        # This function is used to viz the ICP process
        icp = self.update.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(self.update, self.target, max_iter=50)

        # transpose the rotation matrix
        transf_s2t = np.transpose(transf)[0:3, 0:3]

        # update the new model
        self.pcd_update = np.dot(self.pcd_update, transf_s2t)
        self.update = pcl.PointCloud(self.pcd_update.astype(np.float32))
        print(self.pcd_update)

    def VizUpdate(self):

        obj = self.pcd_update
        x_obj = np.asarray(obj [:, 0])
        y_obj = np.asarray(obj [:, 1])
        z_obj = np.asarray(obj [:, 2])
        vtk_update = SEC_VIZ.VtkPointCloud()
        for k in range(len(x_obj)):
            point_obj = ([x_obj[k], y_obj[k], z_obj[k]])
            vtk_update.addPoint(point_obj)
        self.vtk_update = [vtk_update, self.vtk_target]
        print(self.vtk_update)

if __name__ == '__main__':
    test = TestICP()
    test.setUp()
    vtk_source, vtk_target = test.pcd2vtk()
    # SEC_VIZ.vtk_pc([vtk_source,vtk_target])
    # converged, transf, estimate, fitness = test.testICP()

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(400, 400)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Continue iteration
    while(True):
        test.ICPupdate()
        test.VizUpdate()
        actor_1 = test.vtk_update[0].vtkActor
        actor_2 = test.vtk_update[1].vtkActor
        ren.AddActor(actor_1)
        ren.AddActor(actor_2)
        renWin.Render()
        time.sleep(0.1)
        ren.RemoveActor(actor_1)
        ren.RemoveActor(actor_2)
        iren.Initialize()
        renWin.Render()


