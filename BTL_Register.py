
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

import numpy as np
import matplotlib.pyplot as plt
import open3d
import BTL_DataConvert
import math
import BTL_Registration

class Register():

    def __init__(self):

        # Read the scan and brain files
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

        # Define the obj of A and B
        self.A = self.brain_npy
        self.B = self.brain_npy
        affine_paras = [1, 1, 1]
        self.B = TransformImg(self.B, affine_paras)

        # Define the fixed box for each object
        N_A = 10
        N_B = 10
        scale = 2
        self.obj_globalbox = self.A
        self.obj_globalbox[:, 0] = self.obj_globalbox[:, 0] * scale
        self.obj_globalbox[:, 1] = self.obj_globalbox[:, 1] * scale
        self.obj_globalbox[:, 2] = self.obj_globalbox[:, 2] * scale
        self.Box_A = SetGrid(N_A, self.obj_globalbox)
        self.Box_B = SetGrid(N_B, self.obj_globalbox)

    def CostFunction_PC(self, ref, flt, x):

        # Notice the np.exp function here for the problem
        # COSTFun = np.exp(   self.MutualInf(ref,   self.TransformImg(flt, x)   )   )
        COSTFun = (- MutualInf_Pc(self.Box_A, self.Box_B, ref, TransformImg(flt, x)))
        print("The current cost function is ", COSTFun)

        return COSTFun

    def Powell_PC(self, F, x, obj_A, obj_B, h = 0.1, tol = 1.0e-6):

        # Define a new function
        def f(s):
            return F(obj_A, obj_B, x + s * v)

        n = len(x)  # number of design variables
        df = np.zeros(n)  # Decreases of F stored here
        u = np.identity(n)  # Initial vectors here by rows

        for j in range(30):

            xOld = x.copy()  # The input x is usually the xStart point
            fOld = F(obj_A, obj_B, x)  # This is not correct -- F(obj_A, obj_B, xOld)

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

    def TestPcRgs(self):

        # Define the obj
        obj_A = self.A
        obj_B = self.B

        # Optimization
        x = [1, 2, 1]
        self.Powell_PC(self.CostFunction_PC, x, obj_A, obj_B, h = 0.1, tol = 10e-6)

    def TestImgRgs(self):

        # Img registration
        a = 1

def TransformImg(obj, affine_paras):

    # Transform the images from A to B
    # Initial transformation -- the input is a 3D vector

    theta_x = affine_paras[0]
    theta_y = affine_paras[1]
    theta_z = affine_paras[2]

    # Transform the image
    Rform = BTL_DataConvert.AglTransform(theta_x, theta_y, theta_z)
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
    print("The current %s and %s is " % (b, a))
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
    #     nIter = 10

    for i in range(nIter):
        print("The number of nIter is ", nIter)
        print("check search")
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

    # Determine the downhill and change sign if needed
    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = f(x2)

        # check if minimum between x1 - h and x1 + h
        if f2 > f1:
            return x2, x1 - h

    for i in range(100):  # maximum 100 times
        print("check bracket ", i)
        h = c * h
        x3 = x2 + h
        f3 = f(x3)
        print("The f3 is ", f3)
        print("The f2 is ", f2)
        if f3 > f2:
            return x1, x3
        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3

def CostFunction_PC(ref, flt, x):

    # Notice the np.exp function here for the problem
    # COSTFun = np.exp(   self.MutualInf(ref,   self.TransformImg(flt, x)   )   )
    COSTFun =  np.exp( - MutualInf_Pc(Box_A, Box_B, ref, TransformImg(flt, x)))
    print("The current cost function is ", COSTFun)

    return COSTFun

def MutualInf_Pc(Box_A, Box_B, obj_A, obj_B):

    # Test the CAL_Grid
    Grid_A = CAL_Grid(Box_A, obj_A)
    Grid_B = CAL_Grid(Box_B, obj_B)

    # Test the histogram function
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
        Hist[idx] = np.int(len(np.asarray(item_box.points)))

    return Hist

def JointHist(Hist_A, Hist_B):

    # Read the one dimensional histogram
    # fig, axes = plt.subplots(1, 2)

    # The hist for object A
    hist_A = np.histogram(Hist_A, bins = len(Hist_A))
    # axes[0].hist(Hist_A, bins = 20)
    # axes[0].set_title('T1 slice histogram')

    # The hist for object B
    hist_B = np.histogram(Hist_B, bins = len(Hist_B))
    # axes[1].hist(Hist_B, bins = 20)
    # axes[1].set_title('T2 slice histogram')
    # plt.show()

    # Joint histogram -- 2D
    hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
    hist_2d.T[0][0] = 0
    # plt.imshow(hist_2d.T, origin='lower')
    # plt.show()

    return hist_2d

if __name__ == "__main__":

    # test = Register()
    # test.Test()

    test = Register()
    test.TestPcRgs()