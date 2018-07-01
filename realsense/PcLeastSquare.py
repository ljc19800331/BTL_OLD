
# This code targets at the least square solution to find the target R and T
# Algorithm idea:
# show the two coordinate system with 400 plane points
# Add the noise to the ground plane
# Set the truth of the R and T values
# Test the method with ICP and LS solution

import cv2
import vtk
import numpy as np
import DataConvert as DC
import BTL_VIZ as BV
import random
import math
from math import *
import pcl
from numpy.linalg import inv
import LS_Solution

def main():

    # Generate the data sets
    x = np.linspace(0, 20, 20)
    y = np.linspace(0, 20, 20)
    xv, yv = np.meshgrid(x, y)
    x_use = xv.reshape(len(x)*len(y), )
    y_use = yv.reshape(len(x)*len(y), )
    z = np.ones((len(x) * len(y), 1)).reshape(len(x)*len(y),)

    # Add the noise data
    sample = 400
    noise = 0.0008 * np.asarray(random.sample(range(0, 1000), sample))

    # Add the noise to the point cloud
    npy_source = np.zeros((len(x) * len(y), 3))
    npy_source[:,0] = x_use
    npy_source[:,1] = y_use
    npy_source[:,2] = z + noise

    # Define the rotation matrix and translation vector
    theta = [-.031, .4, .59]
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
    npy_target = np.dot(npy_source, transform)
    print(transform)
    print(inv(transform))

    # Viz the result
    # vtk_source = DC.npy2vtk(npy_source)
    # vtk_target = DC.npy2vtk(npy_target)
    # BV.VizVtk([vtk_source, vtk_target])

    # Apply the LS method
    source = pcl.PointCloud(npy_source.astype(np.float32))
    target = pcl.PointCloud(npy_target.astype(np.float32))
    icp = source.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(source, target, max_iter=1000)
    print(transf[0:3,0:3])
    print("Translation: ", transf[3, 0:3])

    # Apply the transform
    npy_result = np.dot(npy_source, inv(transf[0:3,0:3]))

    # Viz the result
    vtk_source = DC.npy2vtk(npy_source)
    vtk_target = DC.npy2vtk(npy_target)
    vtk_result = DC.npy2vtk(npy_result)
    BV.VizVtk([vtk_source, vtk_target, vtk_result])

    # Use the LS solution instead
    R,t = LS_Solution.rigid_transform_3D(npy_source, npy_target)
    print(R)

if __name__ == "__main__":
    main()