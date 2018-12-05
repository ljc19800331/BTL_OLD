
# This is a script for image registration
# Code structure
# Mutual Information
# CostFunction
# TransformImg
# Powell optimization method
# bracket
# Fun for Powell optimization
# GodenSearch
# SetGrid
# CALGrid
# Histogram
# JointHist

# Ref showpair image: https://gist.github.com/jpambrun/0cf01ab512db036ae32a4834a4cd542e
# Ref for the c++ image based registration: https://github.com/ahestevenz/ap-image-register/blob/master/apImageRegister.cpp
# Solution of the 3D point set result:
# 1. The size of the grid
# 2. The object -- new object

# Current problem:
# 1. The optimization process is not converge
# 2. The speed is too slow
# 3. The feature might not be enough

# Possible solution:
# 0. Go over the whole program to see if there is problem
# 1. Random test the parameters -- This is im
# 2. Change the object -- to a tilt plane or a hemisphere -- not working well
# 3. Change the size of grid -- not working well as well
# 4. Change the optimization method -- simplex possible -- not a good idea

# Time:
# 1. Find the related code online
# 2. if not, finish by myself
# 2. Finish the poster if possible
# 3. Find

# figure 1: iteraction VS Metric value -- for 2D image registration

import numpy as np
import matplotlib.pyplot as plt
import open3d
import BTL_DataConvert
import math
import BTL_Registration
import nibabel as nib
import cv2
from rotations import *
from scipy.ndimage import affine_transform
import scipy.io as sio

class Register():

    def __init__(self):

        # Read the scan and brain files
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

        # Define the obj of A and B
        # self.A = self.scan_npy
        # self.B = self.scan_npy
        # affine_paras = [10, 10, -10]    # Basic rigid parameters rotx, roty and rotz
        # self.B = TransformPc(self.B, affine_paras)

        # Define the obj of A and B
        # self.A = self.brain_npy
        # self.B = self.brain_npy
        # affine_paras = [1, 1, 1]
        # self.B = TransformPc(self.B, affine_paras)

        # Read the new point cloud data
        pcd_bunny = open3d.read_point_cloud("/home/mgs/PycharmProjects/BTL_GS/BTL_Data/bunny_315.ply")
        PC_bunny = BTL_DataConvert.Npy2Origin(np.asarray(pcd_bunny.points))
        self.A = PC_bunny
        self.B = PC_bunny
        affine_paras = [5, 5, -5]
        self.B = TransformPc(PC_bunny, affine_paras)

        pcd_A = open3d.PointCloud()
        pcd_B = open3d.PointCloud()
        pcd_A.points = open3d.Vector3dVector(self.A)
        pcd_B.points = open3d.Vector3dVector(self.B)
        # open3d.draw_geometries([pcd_A, pcd_B])

        # Define the fixed box for each object
        N_A = 10
        N_B = 10
        scale = 1.3
        Grid_obj = np.zeros((len(self.A), 3))
        Grid_obj[:, 0] = self.A[:, 0]
        Grid_obj[:, 1] = self.A[:, 1]
        Grid_obj[:, 2] = self.A[:, 2]

        # Grid_obj = self.A
        obj_globalbox = Grid_obj
        obj_globalbox[:, 0] = obj_globalbox[:, 0] * scale
        obj_globalbox[:, 1] = obj_globalbox[:, 1] * scale
        obj_globalbox[:, 2] = obj_globalbox[:, 2] * scale
        self.Box_A = SetGrid(N_A, obj_globalbox)
        self.Box_B = SetGrid(N_B, obj_globalbox)

        # Define the object for application
        self.obj_A = None
        self.obj_B = None

    def CostFunction_PC(self, ref, flt, x):

        # Notice the np.exp function here for the problem
        # COSTFun = np.exp(   self.MutualInf(ref,   self.TransformImg(flt, x)   )   )
        COSTFun = (-MutualInf_Pc(self.Box_A, self.Box_B, ref, TransformPc(flt, x)))
        # print("The current cost function is ", COSTFun)

        return COSTFun

    def CostFunction_Img(self, ref, flt, x):

        COSTFun = (-MutualInf_Img(ref, TransformImg(flt, x)))
        print("The current cost function is ", COSTFun)

        return COSTFun

    def CostFunction_Img3D(self, ref, flt, x):

        COSTFun = (-MutualInf_Img(ref, TransformImg3D(flt, x)))
        # print("The current cost function is ", COSTFun)

        return COSTFun

    def Powell_PC(self, F, x, obj_A, obj_B, h = 0.1, tol = 1.0e-6):

        # Define a new function
        def f(s):
            return F(obj_A, obj_B, x + s * v)

        n = len(x)          # number of design variables
        df = np.zeros(n)    # Decreases of F stored here
        u = np.identity(n)  # Initial vectors here by rows

        for j in range(20):

            print(j)
            # print("The current cost function is ", fOld)

            xOld = x.copy()  # The input x is usually the xStart point
            fOld = F(obj_A, obj_B, x)  # This is not correct -- F(obj_A, obj_B, xOld)
            print("The current cost function is ", fOld)

            # First n line searches record decreases of F -- followed by the last line search algorithm
            for i in range(n):

                # The initial direction on v -- This is im as well
                v = u[i]

                # problem with this cas
                a, b = bracket(f, 0.0, h)

                # For the line search only
                s, fMin = Glodensearch(f, a, b, tol=1.0e-9)
                df[i] = fOld - fMin
                fOld = fMin
                x = x + s * v

            # Last line search in the cycle -- this is im -- why this works and how we can prove that?
            v = x - xOld

            # Calculate the bracket
            a, b = bracket(f, 0.0, h)
            s, fLast = Glodensearch(f, a, b, tol=1.0e-9)
            x = x + s * v

            # Check for convergence
            if math.sqrt(np.dot(x - xOld, x - xOld) / n) < tol:
                return x, j + 1

            # Identify biggest decrease
            iMax = np.argmax(df)

            # update search directions
            for i in range(iMax, n - 1):
                u[i] = u[i + 1]

            u[n - 1] = v

        print("Powell did not converge")
        return x, j + 1

    def Powell_Img(self, F, x, obj_A, obj_B, h = 0.1, tol = 1.0e-6):

        # Define a new function
        def f(s):
            return F(obj_A, obj_B, x + s * v)

        n = len(x)  # number of design variables
        df = np.zeros(n)  # Decreases of F stored here
        u = np.identity(n)  # Initial vectors here by rows

        count_iter = 0

        for j in range(30):

            count_iter += 1

            xOld = x.copy()  # The input x is usually the xStart point
            fOld = F(obj_A, obj_B, x)  # This is not correct -- F(obj_A, obj_B, xOld)

            # print("The current f is ", f)
            print("The current cost function is ", fOld)

            # First n line searches record decreases of F -- followed by the last line search algorithm
            for i in range(n):

                # The initial direction on v -- This is im as well
                v = u[i]

                # problem with this cas
                # print("check ")
                # print("check h", h)
                # print("check f ", f(0.1))
                a, b = bracket(f, 0.0, h)

                # For the line search only
                s, fMin = Glodensearch(f, a, b, tol=1.0e-9)
                df[i] = fOld - fMin
                fOld = fMin
                x = x + s * v

            # Last line search in the cycle -- this is im -- why this works and how we can prove that?
            v = x - xOld

            # Calculate the bracket
            a, b = bracket(f, 0.0, h)
            s, fLast = Glodensearch(f, a, b, tol=1.0e-9)
            x = x + s * v

            # Check for convergence
            if math.sqrt(np.dot(x - xOld, x - xOld) / n) < tol:
                print("The final iteration time is ", count_iter)
                return x, j + 1

            # Identify biggest decrease
            iMax = np.argmax(df)

            # update search directions
            for i in range(iMax, n - 1):
                u[i] = u[i + 1]

            u[n - 1] = v

        print("Powell did not converge")
        # return x, j + 1

    def TestPcRgs(self, obj_A, obj_B):

        pcd_A = open3d.PointCloud()
        pcd_B = open3d.PointCloud()
        pcd_A.points = open3d.Vector3dVector(obj_A)
        pcd_B.points = open3d.Vector3dVector(obj_B)
        # open3d.draw_geometries([pcd_A, pcd_B])

        # Calculate the initial mutual information
        x = [0, 0, 0]
        MI = MutualInf_Pc(self.Box_A, self.Box_B, obj_A, TransformPc(obj_B, x))
        print("The initial mutual information is ", MI)

        # Optimization of the point cloud
        x = [0, 0, 0]
        x_Res, N_iter = self.Powell_PC(self.CostFunction_PC, x, obj_A, obj_B, h = 0.1, tol = 10e-6)
        print("The transformation is ", x_Res)

        # Show the result -- this is im as well
        obj_C = TransformPc(obj_B, x_Res)
        pcd_C = open3d.PointCloud()
        pcd_C.points = open3d.Vector3dVector(obj_C)
        open3d.draw_geometries([pcd_A, pcd_C])

    def TestImgRgs(self):

        # Read the images
        Img_1 = nib.load('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/mni_icbm152_t1_tal_nlin_sym_09a.nii')
        Img_1 = Img_1.get_data()
        Img_1 = Img_1[:, :, 94]
        Img_2 = nib.load('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/mni_icbm152_t2_tal_nlin_sym_09a.nii')
        Img_2 = Img_2.get_data()
        Img_2 = Img_2[:, :, 94]

        rows, cols = Img_1.shape

        # x = [1, 0, 0, 0, 1, 0]
        x = [1, 0, 10, 0, 1, 10]
        M = np.float32([[x[0], x[1], x[2]], [x[3], x[4], x[5]]])
        Img_2 = cv2.warpAffine(Img_2, M, (cols, rows))
        self.obj_A = Img_1
        self.obj_B = Img_2
        show_images([self.obj_A, self.obj_B])
        ShowPair(self.obj_A, self.obj_B)
        # plt.imshow(np.hstack((Img_1, Img_2)))
        # plt.show()

        # Mutual information
        MI = MutualInf_Img(self.obj_A, self.obj_B)
        print("The initial mutual information is ", MI)

        # Optimization
        x = [1, 0, 0, 0, 1, 0]
        x_Res, N_iter = self.Powell_Img(self.CostFunction_Img, x, self.obj_A, self.obj_B, h = 0.1, tol = 10e-6)
        print("The final result is ", x_Res)
        M = np.float32([[x_Res[0], x_Res[1], x_Res[2]], [x_Res[3], x_Res[4], x_Res[5]]])
        Img_out = cv2.warpAffine(self.obj_B, M, (cols, rows))
        ShowPair(self.obj_A, Img_out)
        plt.imshow(np.hstack((self.obj_A, Img_out)))
        plt.show()

    def TestImg3D(self, x_init):

        # Register 3D image with different image modality
        # Load the data
        np.set_printoptions(precision = 4)
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'

        M_x = x_rotmat(0)  # radius 1.57 = 180 degrees
        M_y = y_rotmat(0)
        M_z = z_rotmat(0)
        M = M_x * M_y * M_z
        translation = [0, 0, 0]  # voxel

        img_A = nib.load('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/mni_icbm152_t1_tal_nlin_sym_09a.nii')
        data_A = img_A.get_data()
        Img_A = affine_transform(data_A, M, translation, output_shape = data_A.shape, order = 1)
        I_A = Img_A

        img_B = nib.load('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/mni_icbm152_t2_tal_nlin_sym_09a.nii')
        data_B = img_B.get_data()
        Img_B = affine_transform(data_B, M, translation, output_shape = data_A.shape, order=1)
        I_B = Img_B

        # data_A = sio.loadmat('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/obj_A.mat')
        # I_A = data_A['obj_A']
        #
        # data_B = sio.loadmat('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/obj_B.mat')
        # I_B = data_B['obj_B']

        # print("The shape of the data is ", data.shape)
        # print("The shape of I is ", I.shape)

        M_x = x_rotmat(0)  # radius 1.57 = 180 degrees
        M_y = y_rotmat(0)
        M_z = z_rotmat(0)
        M = M_x * M_y * M_z
        translation = [0, 0, 0]  # voxel
        I_B = affine_transform(I_B, M, translation, output_shape = I_A.shape, order=1)
        obj_A = I_A
        obj_B = I_B
        MI = MutualInf_Img(obj_A, obj_B)
        print("The correct MI is ", MI)

        # Initial transformation
        M_x = x_rotmat(0.1)                 # radius 1.57 = 180 degrees
        M_y = y_rotmat(0.1)
        M_z = z_rotmat(0.2)
        M = M_x * M_y * M_z
        translation = [10, 10, -10]             # voxel
        I_B = affine_transform(I_B, M, translation, output_shape = I_A.shape, order = 1)

        # The mutual information with two 3D images
        obj_A = I_A
        obj_B = I_B
        Hist_A = obj_A.ravel()
        Hist_B = obj_B.ravel()
        print("The histogram of A is ", Hist_A)
        print("The shape of the histogram of A is ", Hist_A.shape)
        MI = MutualInf_Img(obj_A, obj_B)
        hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
        hist_2d.T[0][0] = 0
        # plt.imshow(hist_2d.T, origin='lower')
        # plt.show()
        print("The mutual information between object A and B is ", MI)

        # Optimization
        # x_init = [0.1, 0.1, 0.2, 10, 10, -10]
        self.obj_A = obj_A
        self.obj_B = obj_B
        x_Res, N_iter = self.Powell_Img(self.CostFunction_Img3D, x_init, self.obj_A, self.obj_B, h = 0.1, tol = 10e-6)
        print("The final result is ", x_Res)

        # Show the three images before and after
        obj_C = TransformImg3D(self.obj_B, x_Res)
        obj_use_A = self.obj_A
        obj_use_B = self.obj_B
        obj_use_C = obj_C

        n_x, n_y, n_z = obj_use_A.shape
        Img_test_A = obj_use_A[:, :, n_z // 2]
        Img_test_B = obj_use_B[:, :, n_z // 2]
        Img_test_C = obj_use_C[:, :, n_z // 2]

        # ShowPair(Img_test_A, Img_test_B)
        # ShowPair(Img_test_A, Img_test_C)

def TransformImg3D(Img_3D, affine_paras):

    # affine_paras = [rotx, roty, rotz, tx, ty, tz]

    rotx = affine_paras[0]
    roty = affine_paras[1]
    rotz = affine_paras[2]

    tx = affine_paras[3]
    ty = affine_paras[4]
    tz = affine_paras[5]

    M_x = x_rotmat(rotx)
    M_y = y_rotmat(roty)
    M_z = z_rotmat(rotz)
    M = M_x * M_y * M_z
    translation = [-tx, -ty, -tz]

    Img_out = affine_transform(Img_3D, M, translation, output_shape = Img_3D.shape, order = 1)

    return Img_out

def TransformImg(Img, affine_paras):

    # Transform the image to the new position
    rows, cols = Img.shape
    a11 = affine_paras[0]
    a12 = affine_paras[1]
    tx = affine_paras[2]
    a21 = affine_paras[3]
    a22 = affine_paras[4]
    ty = affine_paras[5]

    M = np.float32([[a11, a12, tx], [a21, a22, ty]])
    Img_out = cv2.warpAffine(Img, M, (cols, rows))

    return Img_out

def MutualInf_Img(obj_A, obj_B):

    # Get the histogram image
    Hist_A = obj_A.ravel()
    Hist_B = obj_B.ravel()

    # The joint histogram
    Hist_2d = JointHist(Hist_A, Hist_B)
    pxy = Hist_2d / float(np.sum(Hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def TransformPc(obj, affine_paras):

    # Transform the images from A to B
    # Initial transformation -- the input is a 3D vector

    theta_x = affine_paras[0]
    theta_y = affine_paras[1]
    theta_z = affine_paras[2]

    # Transform the image
    Rform = BTL_DataConvert.AglTransform(theta_x, theta_y, theta_z)
    # print("The transform is ", Rform)
    obj_tformed = np.matmul(obj, Rform)

    # Set to the origin -- this is im as well
    obj_tformed = np.asarray(BTL_DataConvert.Npy2Origin(obj_tformed))

    return obj_tformed

def Powell(F, x, h = 0.1, tol = 1.0e-6):

    # Define a new function
    def f(s):
        return F(x + s * v)

    n = len(x)  # number of design variables
    df = np.zeros(n)  # Decreases of F stored here
    u = np.identity(n)  # Initial vectors here by rows

    for j in range(30):

        print(j)

        # Allow for only 30 cycles (loops) - maximum n times -- no less then 20 is good enough
        # j is the number of iteration -- this is a good idea

        xOld = x.copy()     # The input x is usually the xStart point
        print("The xOld is ", xOld)
        print("The current Function is ", F)
        fOld = F(x)      # This is not correct -- F(obj_A, obj_B, xOld)

        # First n line searches record decreases of F -- followed by the last line search algorithm
        for i in range(n):

            print("check point")

            # The initial direction on v -- This is im as well
            v = u[i]

            # problem with this cas
            a, b = bracket(f, 0.0, h)

            # For the line search only
            s, fMin = Glodensearch(f, a, b, tol=1.0e-9)
            df[i] = fOld - fMin
            fOld = fMin
            x = x + s * v

        # Last line search in the cycle -- this is im -- why this works and how we can prove that?
        v = x - xOld

        # Calculate the bracket
        a, b = bracket(f, 0.0, h)
        s, fLast = Glodensearch(f, a, b, tol=1.0e-9)
        x = x + s * v

        # Check for convergence
        if math.sqrt(np.dot(x - xOld, x - xOld) / n) < tol:
            return x, j + 1

        # Identify biggest decrease
        iMax = np.argmax(df)

        # update search directions
        for i in range(iMax, n - 1):
            u[i] = u[i + 1]

        u[n - 1] = v

    print("Powell did not converge")

def Glodensearch(f, a, b, tol = 1.0e-9):

    # Initial position
    # Calculate the total iteration numbers
    # print("The current %s and %s is " % (b, a))
    nIter = int(math.ceil(-2.078087 * math.log(tol / abs(b - a))))
    R = 0.618033989
    C = 1.0 - R

    # First telescoping -- Define the golden length
    x1 = R * a + C * b
    x2 = C * a + R * b
    f1 = f(x1)
    f2 = f(x2)

    # Main loop
    # if nIter > 10:
    #     nIter = 20

    for i in range(nIter):
        # print("The number of nIter is ", nIter)
        # print("check search")
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = C * a + R * b
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = R * a + C * b
            f1 = f(x1)
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

def bracket(f, x1, h):

    # Find the search range only based on the x1, h and f (function)
    # Find the bracket within this function
    c = 1.618033989

    # print("The f inside the function is ", f)
    f1 = f(x1)
    x2 = x1 + h
    f2 = f(x2)
    # print("The current f1 is ", f1)
    # print("The current f2 is ", f2)

    # Determine the downhill and change sign if needed
    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = f(x2)

        # check if minimum between x1 - h and x1 + h
        if f2 > f1:
            return x2, x1 - h

    for i in range(100):  # maximum 100 times
        # print("check bracket ", i)
        h = c * h
        x3 = x2 + h
        f3 = f(x3)
        # print("The f3 is ", f3)
        # print("The f2 is ", f2)
        if f3 > f2:
            return x1, x3
        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3

    # return x1, x3

def CostFunction_PC(ref, flt, x):

    # Notice the np.exp function here for the problem
    # COSTFun = np.exp(   self.MutualInf(ref,   self.TransformImg(flt, x)   )   )
    COSTFun =  np.exp( - MutualInf_Pc(Box_A, Box_B, ref, TransformImg(flt, x)))
    print("The current cost function is ", COSTFun)

    return COSTFun

def MutualInf_Pc(Box_A, Box_B, obj_A, obj_B):

    # Test the CAL_Grid
    Grid_A = CAL_Grid_1(Box_A, obj_A)
    Grid_B = CAL_Grid_1(Box_B, obj_B)

    # print("The grid of A is ", Grid_A)
    # print("The grid of B is ", Grid_B)

    # Test the histogram function
    # Hist_A = Histogram(Grid_A)
    # Hist_B = Histogram(Grid_B)

    Hist_A = Histogram(Grid_A)
    Hist_B = Histogram(Grid_B)

    # The joint histogram -- 1D
    Hist_2d = JointHist(Hist_A, Hist_B)

    # Mutual information for joint histogram -- The result is a 2D matrix
    pxy = Hist_2d / float(np.sum(Hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x

    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def SetGrid(N, obj):

    # obj = self.obj_globalbox
    # The initial definition of our parameter setting
    x_min = np.min(obj[:, 0])
    x_max = np.max(obj[:, 0])
    y_min = np.min(obj[:, 1])
    y_max = np.max(obj[:, 1])
    z_min = np.min(obj[:, 2])
    z_max = np.max(obj[:, 2])

    # 3D meshgrid
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    z = np.linspace(z_min, z_max, N)

    # 3D box Creation -- save the index in the box
    Dict = np.zeros([(len(x) - 1) * (len(y)-1) * (len(z)-1), 7])

    print("The shape of the dictionary is ", Dict.shape)

    count = 0
    for idx_x in range(len(x) - 1):
        x1 = x[idx_x]
        x2 = x[idx_x + 1]
        for idx_y in range(len(y) - 1):
            y1 = y[idx_y]
            y2 = y[idx_y + 1]
            for idx_z in range(len(z) - 1):
                # print(count)
                z1 = z[idx_z]
                z2 = z[idx_z + 1]
                IDX = np.asarray([idx_x, idx_y, idx_z])
                Dict[count, :] = np.asarray([x1, x2, y1, y2, z1, z2, count])
                count += 1

    return Dict

def CAL_Grid(Box, obj_npy):

    # Define the GRID object to save the voxel point sets
    GRID = []

    # Plot the grid in different ways
    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

    # Get the box within this region
    pcdlist = [mesh_frame]
    pcdlist_nonempty = [mesh_frame]

    for i in range(len(Box)):

        box = Box[i, :]
        pcd = Get_Grid(box, obj_npy)
        # Save only the non-empty region
        pcdlist.append(pcd)
        if len(np.asarray(pcd.points)) != 0:
            pcdlist_nonempty.append(pcd)

    # Save only the grid regions
    GRID = pcdlist[1:]
    # print("The length of non empty girds are ", len(pcdlist_nonempty))

    return GRID

def CAL_Grid_1(Box, obj_npy):

    # Calculate the feature of variance of the z values
    # Define the GRID object to save the voxel point sets
    GRID = []

    # Plot the grid in different ways
    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

    # Get the box within this region
    pcdlist = [mesh_frame]
    pcdlist_nonempty = [mesh_frame]

    for i in range(len(Box)):

        box = Box[i, :]
        Var= Get_Variance(box, obj_npy)
        # Save only the non-empty region
        pcdlist.append(Var)
        if Var != 0:
            pcdlist_nonempty.append(Var)

    # print("The pcdlist is ", pcdlist)

    # Save only the grid regions
    GRID = pcdlist[1:]


    return GRID

def Get_Variance(box, obj_npy):

    # Get the variance of all the points
    # Load the brain model
    model = obj_npy

    # Read the box boundary and return the index
    idx = np.where((model[:, 0] > box[0]) & (model[:, 0] < box[1]) \
                   & (model[:, 1] > box[2]) & (model[:, 1] < box[3]) \
                   & (model[:, 2] > box[4]) & (model[:, 2] < box[5]))

    # Convert to the pcd file
    obj = model[idx, :]
    obj = obj.reshape((len(idx[0]), 3))

    # Get variance of the points -- z heights
    Z = obj[:, 2]   # The Z height values
    Var = np.var(Z)

    return Var

def Get_Grid(box, obj_npy):

    # Load the brain model
    model = obj_npy

    # Read the box boundary and return the index
    idx = np.where( (model[:, 0] > box[0]) & (model[:, 0] < box[1])  \
                    & (model[:, 1] > box[2]) & (model[:, 1] < box[3]) \
                    & (model[:, 2] > box[4]) & (model[:, 2] < box[5]) )

    # Convert to the pcd file
    obj = model[idx, :]
    obj = obj.reshape((len(idx[0]), 3))
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(obj)

    return pcd

def Histogram(GRID):

    # Show the intensity with the voxel ID
    Hist = np.random.normal(size = len(GRID))

    for idx, item_box in enumerate(GRID):
        if item_box == np.nan:
            item_box = 0
        # print(np.nan_to_num(item_box))
        # print(item_box)
        # print(item_box == np.NaN)
        Hist[idx] = np.float(np.nan_to_num(item_box))   # np.int(item_box)
        # Hist[idx] = np.int(len(np.asarray(item_box.points)))
    return Hist

def JointHist(Hist_A, Hist_B):

    # Read the one dimensional histogram
    # fig, axes = plt.subplots(1, 2)

    # The hist for object A
    # hist_A = np.histogram(Hist_A.ravel(), bins = len(Hist_A))
    # axes[0].hist(Hist_A, bins = 20)
    # axes[0].set_title('T1 slice histogram')

    # The hist for object B
    # hist_B = np.histogram(Hist_B.ravel(), bins = len(Hist_B))
    # axes[1].hist(Hist_B, bins = 20)
    # axes[1].set_title('T2 slice histogram')
    # plt.show()

    # Joint histogram -- 2D
    hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
    # hist_2d.T[0][0] = 0
    # plt.imshow(hist_2d.T, origin='lower')
    # plt.show()

    return hist_2d

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def ShowPair(Img_1, Img_2):
    # img = cv2.imread('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_cortex.jpg',0)
    rows, cols = Img_1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)
    # img2 = cv2.warpAffine(img,M,(cols,rows))
    diff = ((Img_1.astype(np.int16) - Img_2.astype(np.int16)) / 2 + 128).astype(np.uint8)
    im_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    lut = np.zeros((256, 1, 3), dtype="uint8")
    for i in range(256):
        # print(i)
        lut[i, 0, 0] = max(min((127 - i) * 2, 255), 0) if i < 127 else 0
        lut[i, 0, 1] = max(min((i - 127) * 2, 255), 0) if i > 127 else 0
        lut[i, 0, 2] = max(min((127 - i) * 2, 255), 0) if i < 127 else 0

    im_falsecolor_r = cv2.LUT(diff, lut[:, 0, 0])
    im_falsecolor_g = cv2.LUT(diff, lut[:, 0, 1])
    im_falsecolor_b = cv2.LUT(diff, lut[:, 0, 2])
    im_falsecolor = np.dstack((im_falsecolor_r, im_falsecolor_g, im_falsecolor_b))

    plt.imshow(im_falsecolor)
    plt.show()

    # cv2.imshow('figure5', im_falsecolor)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    # test = Register()
    # test.Test()

    # Test the 2D image
    # test = Register()
    # test.TestImgRgs()

    # Test the 2D image
    test = Register()

    X_init = [[0.1, 0.1, 0.1, 0, 0, 0],
              [0.2, 0.2, 0.2, 0, 0, 0],
              [0.5, 0.5, 0.5, 0, 0, 0],
              [0.1, 0.1, 0.1, 10, 20, 30],
              [0.2, 0.2, 0.2, -10, -10, -10],
              [0.1, 0.1, 0.1, 10, 10, 10],
              [0, 0, 0, 50, 50, -50],
              [0, 0, 0, 10, 10, -10],
              [0, 0, 0, -20, 20, 20],
              [0, 0, 0, 30, 30, 30]]

    for item in X_init:
        print("The current initial condition is ", item)
        test.TestImg3D(item)

    # x_init = [0.1, 0.1, 0.2, 10, 10, -10]


    # Test the 3D image
    # test = Register()
    # obj_A = test.A
    # obj_B = test.B
    # test.TestPcRgs(obj_A, obj_B)