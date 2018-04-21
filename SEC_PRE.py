# This function is used for the data preprocessing
# structure: standard format -- npy
# txt2vtk:
# vec2mat:
# mat2vec:
# npy2vtk:
# vtkAngle: transform the point sets based on the x,y,z angles
# This module includes all the format exchange

import numpy as np
from stl import mesh
import SEC_VIZ

def txt2vtk(txt_x, txt_y, txt_z):

        # input: 3D txt data
        # output: vtk data format

        # initialize
        a_x = []; a_y = []; a_z = []
        scan_x = []; scan_y = []; scan_z = []

        # Load the data
        scan_L1x = open(txt_x, "r")
        for line in scan_L1x:
            a_x = line.split()
            scan_x.append(float(a_x[0]))

        scan_L1y = open(txt_y, "r")
        for line in scan_L1y:
            a_y = line.split()
            scan_y.append(float(a_y[0]))

        scan_L1z = open(txt_z, "r")
        for line in scan_L1z:
            a_z = line.split()
            scan_z.append(float(a_z[0]))

        # Normalize the data
        data = scan_x
        data_min = [min(scan_x)] * len(scan_x)
        scan_x = [data - data_min for data, data_min in zip(data, data_min)]
        data = scan_y
        data_min = [min(scan_y)] * len(scan_y)
        scan_y = [data - data_min for data, data_min in zip(data, data_min)]
        data = scan_z
        data_min = [min(scan_z)] * len(scan_z)
        scan_z = [data - data_min for data, data_min in zip(data, data_min)]

        # close the txt files
        scan_L1x.close(); scan_L1y.close(); scan_L1z.close()

        # viz part
        # convert to the point cloud -- vtk format
        vtk_scan = SEC_VIZ.VtkPointCloud()
        for k in range(len(scan_x)):
            point = ([scan_x[k], scan_y[k], scan_z[k]])
            vtk_scan.addPoint(point)

        return vtk_scan, scan_x, scan_y, scan_z

def vec2mat(npy_x, npy_y, npy_z):

    npy_mat = np.zeros((len(npy_x), 3), dtype=np.float32)
    for i in range(0, len(npy_x)):
        npy_mat[i][0] = npy_x[i]
        npy_mat[i][1] = npy_y[i]
        npy_mat[i][2] = npy_z[i]
    return npy_mat

def mat2vec(npy_mat):
    py_mat = np.zeros((len(npy_mat[:, 0]), 3), dtype=np.float32)
    npy_x = npy_mat[:, 0]
    npy_y = npy_mat[:, 1]
    npy_z = npy_mat[:, 2]
    return npy_x, npy_y, npy_z

def npy2vtk(npy_data):

    vtk_data = SEC_VIZ.VtkPointCloud()
    x = np.asanyarray(npy_data[:,0])
    y = np.asanyarray(npy_data[:,1])
    z = np.asanyarray(npy_data[:,2])
    for k in range(len(x)):
        point = ([x[k], y[k], z[k]])
        #print(point)
        vtk_data.addPoint(point)
    return vtk_data

if __name__ == '__main__':
    txt_x = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_x.txt'
    txt_y = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_y.txt'
    txt_z = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_z.txt'

    # test txt2vtk
    vtk_scan, scan_x, scan_y, scan_z = txt2vtk(txt_x, txt_y, txt_z)
    print(scan_x[1:2]); print('\n')
    print(scan_y[1:2]); print('\n')
    print(scan_z[1:2])
    SEC_VIZ.vtk_pc([vtk_scan])

    # test vec2mat

    # test npy2vtk

