# Data convertion
# All functions without class structure
# txt2vtk: txt file to vtk file
# vec2mat: npy vector to matrix
import numpy as np
import math
import plyfile
from plyfile import PlyData
import BTL_VIZ
from BTL_VIZ import *

def txt2vtk(txt_x, txt_y, txt_z):

    # initialize
    a_x = []; a_y = []; a_z = [];
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
    vtk_scan = BTL_VIZ.VtkPointCloud()
    for k in range(len(scan_x)):
        point = ([scan_x[k], scan_y[k], scan_z[k]])
        vtk_scan.addPoint(point)

    return vtk_scan, scan_x, scan_y, scan_z

def vec2mat(npy_x, npy_y, npy_z):

    # convert the single vector to the matrix
    npy_mat = np.zeros((len(npy_x), 3), dtype=np.float32)
    npy_mat[:,0] = npy_x
    npy_mat[:,1] = npy_y
    npy_mat[:,2] = npy_z
    return npy_mat

    # for i in range(0, len(npy_x)):
    #     npy_mat[i][0] = npy_x[i]
    #     npy_mat[i][1] = npy_y[i]
    #     npy_mat[i][2] = npy_z[i]
    # return npy_mat

def npy2vtk(npy_data):

    vtk_data = BTL_VIZ.VtkPointCloud()
    x = np.asanyarray(npy_data[:,0])
    y = np.asanyarray(npy_data[:,1])
    z = np.asanyarray(npy_data[:,2])
    for k in range(len(x)):
        point = ([x[k], y[k], z[k]])
        #print(point)
        vtk_data.addPoint(point)
    return vtk_data

def mat2vec(npy_mat):

    # convert the npy matrix to the single vector
    npy_x = npy_mat[:, 0]
    npy_y = npy_mat[:, 1]
    npy_z = npy_mat[:, 2]
    return npy_x, npy_y, npy_z

def AglTransform(Agl_x, Agl_y, Agl_z):

    # Rotary transform in 3D space
    # Input: angle degree (ex: 90, 180, 45)
    # Output: 3 by 3 rotation matrix
    theta_x = math.pi / 180 * Agl_x
    theta_y = math.pi / 180 * Agl_y
    theta_z = math.pi / 180 * Agl_z
    Rx = np.matrix([[1, 0, 0], [0, math.cos(theta_x), math.sin(theta_x)], [0, -math.sin(theta_x), math.cos(theta_x)]])
    Ry = np.matrix([[math.cos(theta_y), 0, -math.sin(theta_y)], [0, 1, 0], [math.sin(theta_y), 0, math.cos(theta_y)]])
    Rz = np.matrix([[math.cos(theta_z), math.sin(theta_z), 0], [-math.sin(theta_z), math.cos(theta_z), 0], [0, 0, 1]])
    R_tform = Rx * Ry * Rz
    return R_tform

def Npy2Origin(npy_model):

    # move all the point to the origin
    npy_model[:,0] = npy_model[:,0] - np.min(npy_model[:,0])
    npy_model[:,1] = npy_model[:,1] - np.min(npy_model[:,1])
    npy_model[:,2] = npy_model[:,2] - np.min(npy_model[:,2])

    return npy_model

def Ply2Npy(filename):

    filename = 'test.ply'
    plydata = PlyData.read('test.ply')
    print(plydata)
    # Get the points of the ply files
    x_ply = plydata['vertex']['x']
    y_ply = plydata['vertex']['y']
    z_ply = plydata['vertex']['z']
    data_ply = np.zeros((len(x_ply), 3), dtype=float)
    data_ply[:, 0] = x_ply
    data_ply[:, 1] = y_ply
    data_ply[:, 2] = z_ply
    npy_pt = data_ply
    return npy_pt

def Pc2Origin(npy_model):

    npy_obj = np.zeros( ( len(npy_model[:,0]), 3  ), float)
    npy_obj[:,0] = npy_model[:,0] - np.min(npy_model[:,0])
    npy_obj[:,1] = npy_model[:,1] - np.min(npy_model[:,1])
    npy_obj[:,2] = npy_model[:,2] - np.min(npy_model[:,2])

    return npy_obj

def PcScale(npy_brain, x_max, y_max, z_max):
    # x_max = 3.346
    # y_max = 4.547
    # z_max = 1.571
    npy_use = np.zeros( (len(npy_brain[:,0]) ,3) , float   )
    npy_use[:,0] = npy_brain[:,0] * x_max / np.max(npy_brain[:,0])
    npy_use[:,1] = npy_brain[:,1] * y_max / np.max(npy_brain[:,1])
    npy_use[:,2] = npy_brain[:,2] * z_max / np.max(npy_brain[:,2])
    return npy_use

def PtRegion(npy_data, x_range, y_range):

    # x_center = 1.673; y_center = 2.323; range = 1.0
    # x_range = [(x_center - range/2), (x_center + range/2)]
    # y_range = [(y_center - range/2), (y_center + range/2)]

    # Get the range values
    x_min = x_range[0]; x_max = x_range[1]
    y_min = y_range[0]; y_max = y_range[1]

    # Set the range for x
    x_idx = np.transpose(  np.array(np.where(  npy_data[:,0] > x_range[0]  ))   )
    x_obj = np.reshape(  npy_data[x_idx,:]  , (len(x_idx), 3)  )
    x_idx = np.transpose(  np.array(np.where(  x_obj[:,0] < x_range[1]  ))   )
    x_obj = np.reshape(  x_obj[x_idx,:]  , (len(x_idx), 3)  )
    x_test = npy2vtk(x_obj[:, 0], x_obj[:, 1], x_obj[:, 2])

    # Set the range for y
    y_idx = np.transpose(np.array(np.where(x_obj[:, 1] > y_range[0])))
    y_obj = np.reshape(  x_obj[y_idx, :], (len(y_idx), 3))
    y_idx = np.transpose(np.array(np.where(y_obj[:, 1] < y_range[1])))
    y_obj = np.reshape(y_obj[y_idx, :], (len(y_idx), 3))

    y_test = npy2vtk(y_obj[:, 0], y_obj[:, 1], y_obj[:, 2])

    return y_obj

def pixel2Len(p_center, img_brain, flag_type, model_Len):

    height, width = img_brain.shape[:2]

    if flag_type == 'scan':
        Len_x = np.float(p_center[0]) / width * model_Len[0]
        Len_y = np.float(p_center[1]) / height * model_Len[1]

    return [Len_x, Len_y]

def pt2Len(pt_npy, flag_type, flag_Len):

    pt_xyz = pt_npy

    pt_xyz[:, 0] = pt_xyz[:,0] -  np.min(pt_xyz[:,0])
    pt_xyz[:, 1] = pt_xyz[:,1] -  np.min(pt_xyz[:,1])
    pt_xyz[:, 2] = pt_xyz[:,2] -  np.min(pt_xyz[:,2])

    x_range = np.max(pt_xyz[:,0]) - np.min(pt_xyz[:,0])
    y_range = np.max(pt_xyz[:,1]) - np.min(pt_xyz[:,1])
    z_range = np.max(pt_xyz[:,2]) - np.min(pt_xyz[:,2])

    if flag_type == 'brain':

        if flag_Len == 'cm':
            x_len = 8.5
            y_len = 11.8
            z_len = 4.19
            c_x = x_len / x_range
            c_y = y_len / y_range
            c_z = z_len / z_range
            pt_xyz[:,0] = pt_xyz[:,0] * c_x
            pt_xyz[:,1] = pt_xyz[:,1] * c_y
            pt_xyz[:,2] = pt_xyz[:,2] * c_z
        elif flag_Len == 'inch':
            x_len = 3.346
            y_len = 4.646
            z_len = 1.65
            c_x = x_len / x_range
            c_y = y_len / y_range
            c_z = z_len / z_range
            pt_xyz[:, 0] = pt_xyz[:, 0] * c_x
            pt_xyz[:, 1] = pt_xyz[:, 1] * c_y
            pt_xyz[:, 2] = pt_xyz[:, 2] * c_z

    if flag_type == 'scan':

        if flag_Len == 'cm':
            c = 2.54
            pt_xyz[:,0] = pt_xyz[:,0] * c
            pt_xyz[:,1] = pt_xyz[:,1] * c
            pt_xyz[:,2] = pt_xyz[:,2] * c

        elif flag_Len == 'inch':
            c = 1
            pt_xyz[:,0] = pt_xyz[:,0] * c
            pt_xyz[:,1] = pt_xyz[:,1] * c
            pt_xyz[:,2] = pt_xyz[:,2] * c

    return pt_xyz

def Readtxt(txt_path):

    data = []
    #txt_path = 'brain_Fx.txt'
    txt_file = open(txt_path, "r")
    for line in txt_file:
        a_x = line.split()
        data.append(float(a_x[0]))
    data = np.array(data)

    return data

def list2pt(list):

    # This function aims to convert the list to the point values
    X = np.zeros((len(list), 1))
    Y = np.zeros((len(list), 1))
    for i in range(len(list)):
        X[i] = list[i][0][0]
        Y[i] = list[i][0][1]

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X,Y

def ind2sub(idx, h, w):
    # input: the index of the pixel in the image
    # output: the specific subscript
    row = int(np.round(idx / w))
    col = idx - row * w
    return row, col

def sub2ind(row, col, h, w):
    # input: the array of row and column
    # output: the index of the specific subscript
    idx = (row - 1) * w + col
    return idx

if __name__ == '__main__':

    # Test txt2vtk
    txt_x = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_x.txt'
    txt_y = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_y.txt'
    txt_z = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_z.txt'
    vtk_scan, scan_x, scan_y, scan_z = txt2vtk(txt_x, txt_y, txt_z)
    VizVtk([vtk_scan])

    # Test vec2mat(npy_data):
    npy_mat = vec2mat(scan_x, scan_y, scan_z)
    print(npy_mat.shape)
    print(npy_mat)

    # Test npy2vtk
    vtk_data = npy2vtk(npy_mat)
    VizVtk([vtk_data])

    # Test mat2vec
    npy_x, npy_y, npy_z = mat2vec(npy_mat)
    print(npy_x.shape)
    print(npy_x)

    # Test Pc2Origin
    print(npy_mat)
    npy_origin = Pc2Origin(npy_mat)
    print(npy_origin)