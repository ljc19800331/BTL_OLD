# The mutual information based registration
# The is based on the paper: Robust and Fast 3D Scan Alignment using Mutual Information

# Algorithm
# 0. Devide the 3D point sets with 3D grid volume (define a radius and patches) -- This is im
# 1. Voxel feature from 3D point sets
# 2. Encode the 3D feature into the histogram book
# 3. Determine two random variables
# 4. Mutual information based registration
# 5. Optimization algorithms
# 6. Why FPFH/PPF is not working?

# 7. How to do it in realtime? -- how fast it is for the global registration? Maybe using AI? -- This is an interesting question
# 8. MI is good that we don't need to find the patches for matching
# 9. How to design the 3D descriptor such that it can be used for other matching -- point cloud downsample and others
# 10. PC resolution recontruction is a really important steps
# 11. Blank and unoccupied voxel should also be matched in the same time
# 12. X and Y is the modality feature of the two point cloud -- This is im as well
# 13. Define a subregion only for the overlapping region only such that we can reduce the computational time
# 14. How about we weight the features between the two choices and compare only one feature ? - This is worth trying
# 15. Look into the image registration process -- This is im as well
# 16. Idea: t1_slice.ravel() -- This is from the image and we can construct any features based on this idea
# 17. Poweller's method + Simplex method for optimization

# Optimization -- This is the most difficult part
# Cost function -- MI based cost function
# method 1: Simplex method -- how to solve the local minimum problem ? -- Think about that
# method 2: Simplex method -- how to

# Golden search: http://dxyq.tykj.gov.cn/kuai_su/youhuasheji/suanfayuanli/2.2.asp
# Powell's search method
# A good reference for the Powell's method

# Test metrics:
# 1. ICP -- Test and compare the algorithm
# 2. GOICP
# 3. Test other methods

# Reference:
# 1. https://github.com/mariusherzog/ImageRegistration
# 2. Some good reference with the code implementation -- https://github.com/ahestevenz/ap-image-register
# 2. Parzen windows -- used for this idea https://blog.csdn.net/angel_yuaner/article/details/47951111
# 3. im information for image registration: http://pyimreg.github.io/
# 4. Good for learning image related knowledge
# 5. https://github.com/mariusherzog/ImageRegistration -- im inference for this project
# 6. Possible idea for the code implementation https://github.com/spm/spm12/blob/master/spm_coreg.m

# Design the experiment
# 1. object with the model scan data
# 2. gradient free optimization method
# 3. The order of the ID for showing the histogram -- this is im for 3D histogram as well

# The test metrics:
# 1. Crop the PC region -- as the scan model first
# 2. Test with different scanning model
# 3. Test with different parameters or optimization methods

# The algorithms:
# 0. This is a good reason for the case
# 1. Data preprocessing module -- define X and Y grid together
#       1.1 init function
#       1.2 DataUpsamling function
#       1.3 Find an easy case
# 2. Optimization module -- powell method -- pending
#       2.1 Powell method
# 3. Cost function definition module
# 4. Mutual information module
# 5. Registration process module

# Specific implementation:
# 1. GridPatch -- Design the grid for the volume -- finished
    # Label the voxel ID -- finished
    # Set the default grid size -- bound region size
# 2. 3D Feature -- Calculate the 3D feature within the 3D voxel -- This is 1D feature
    # The number of points within the grid
    # The order of the voxel grid
    # The variance of the 3D point sets within the grid
    # The histogram corresponding to the voxel ID
# 3. Mutual information -- Calculate the mutual information based on the joint histogram
    # The transform function
# 4. Register voxelized images

import open3d
import numpy as np
import BTL_DataConvert
import BTL_MAP
import math
from open3d import *
import numpy as np
# from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import BTL_Registration
import open3d
import matplotlib.pyplot as plt
import BTL_DataConvert

class Register():

    def __init__(self):

        # Define the classes
        self.MIR = MIR()
        self.PatchGrid = PatchGrid()

        # Get the Histogram for both cases
        # Get the box for object A, B
        N_A = 10    # The size of the grid
        N_B = 10    # The same as before
        self.Box_A = self.PatchGrid.SetGrid(N_A, self.PatchGrid.obj_globalbox)
        self.Box_B = self.PatchGrid.SetGrid(N_B, self.PatchGrid.obj_globalbox)

        # Test the CAL_Grid
        self.Grid_A = self.PatchGrid.CAL_Grid(self.Box_A, self.PatchGrid.A)
        self.Grid_B = self.PatchGrid.CAL_Grid(self.Box_B, self.PatchGrid.B)

        # Test the histogram function
        self.Hist_A = self.PatchGrid.Histogram(self.Grid_A)
        self.Hist_B = self.PatchGrid.Histogram(self.Grid_B)

        # The joint histogram -- 1D
        self.hist_2d = self.PatchGrid.JointHist(self.Hist_A, self.Hist_B)

        # Show the two pcd objects
        # self.PatchGrid.PlotPcdlist([self.PatchGrid.A, self.PatchGrid.B])

    def MutualInf_Img(self):

        # Mutual information for 2D image only
        a = 1

        # This is a nice
        a = 1

    def MutualInf(self, obj_A, obj_B):

        # The joint histogram is a 2D case
        # The input is the image object
        # Suppose our box is big enough such that it can cover all the transformation boundary

        # Test the CAL_Grid
        Grid_A = self.PatchGrid.CAL_Grid(self.Box_A, obj_A)
        Grid_B = self.PatchGrid.CAL_Grid(self.Box_B, obj_B)

        # Test the histogram function
        Hist_A = self.PatchGrid.Histogram(Grid_A)
        Hist_B = self.PatchGrid.Histogram(Grid_B)

        # The joint histogram -- 1D
        Hist_2d = self.PatchGrid.JointHist(Hist_A, Hist_B)

        # Mutual information for joint histogram -- The result is a 2D matrix
        pxy = Hist_2d / float(np.sum(Hist_2d))
        px = np.sum(pxy, axis=1)                    # marginal for x over y
        py = np.sum(pxy, axis=0)                    # marginal for y over x

        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0

        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def CostFunction(self, ref, flt, x):

        # Notice the np.exp function here for the problem
        # COSTFun = np.exp(   self.MutualInf(ref,   self.TransformImg(flt, x)   )   )

        COSTFun =  np.exp(-self.MutualInf(ref, self.TransformImg(flt, x)))
        print("The current cost function is ", COSTFun)

        return COSTFun

    def TransformImg(self, obj, affine_paras):

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

    def Test(self):

        # MI before optimization
        obj_A = self.PatchGrid.A
        obj_B = self.PatchGrid.B
        MI = self.MutualInf(obj_A, obj_B)
        print("The MI before transformation is ", MI)

        # Optimization
        x = [1, 2, 1]       # Initial values of x
        # COSTFun = self.CostFunction(obj_A, obj_B, x)
        self.MIR.Powell(self.CostFunction, x, obj_A, obj_B, h = 0.1, tol = 10e-6)

        # The input vector
        # print("The MI after transformation is ", COSTFun)

class MIR():

    def __init__(self):

        # Load the scan data
        self.filename_scanB = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/scan.pcd'

        # Load the scanA data
        self.scanA = open3d.read_point_cloud(self.filename_scanB)
        self.scanA = np.asarray(self.scanA.points)

        # Load the scanB data
        self.scanB = open3d.read_point_cloud(self.filename_scanB)
        self.scanB = np.asarray(self.scanB.points)

        # Apply the initial transformation
        theta_x = 90
        theta_y = 0
        theta_z = 0
        T_init = BTL_DataConvert.AglTransform(theta_x, theta_y, theta_z)    # initial transformation
        self.scanB = np.matmul(self.scanA, T_init)

        # Center the two scan point cloud
        self.scanA = BTL_DataConvert.Npy2Origin(self.scanA)
        self.scanB = BTL_DataConvert.Npy2Origin(self.scanB)
        self.scanB = np.asarray(self.scanB)

        # print(self.scanB.shape)
        # Show the point cloud together
        self.pcd_scanA = open3d.PointCloud()
        self.pcd_scanA.points = open3d.Vector3dVector(self.scanA)
        self.pcd_scanB = open3d.PointCloud()
        self.pcd_scanB.points = open3d.Vector3dVector(self.scanB)

        # Resample the PC such that it can be used for registration

        # Viz the final results
        coordinate_frame = open3d.create_mesh_coordinate_frame(size = 5, origin=[0, 0, 0])
        # open3d.draw_geometries([self.pcd_scanA, self.pcd_scanB, coordinate_frame])

    def DataUpsample(self):

        # This function is used to upsample the point cloud such that it can be used for registration
        # Read the scan image

        # Scan depth image
        npy_scan = self.scanA
        nx = 200
        ny = 200
        theta_x = 0
        theta_y = 0
        theta_z = 0
        theta = [theta_x, theta_y, theta_z]
        BTLMAP = BTL_MAP.AutoPcMap()
        _ , img_scan, [grid_x, grid_y, grid_z] = BTLMAP.Scan2img(npy_scan, nx, ny, theta)

        # Show the grid_z
        print("The grid_x is ", grid_x.shape)
        print("The grid y is ", grid_y.shape)
        print("The grid z is ", grid_z)

        # Scan depth to point cloud
        # Depth image from point cloud
        depth_img = np.asarray(img_scan)
        # depth_img = np.asarray(grid_z)

        print(depth_img.shape)

        Range_all = [np.min(npy_scan[:, 0]), np.max(npy_scan[:, 0]), np.min(npy_scan[:, 1]), np.max(npy_scan[:, 1]),
                     np.min(npy_scan[:, 2]), np.max(npy_scan[:, 0])]

        PC_depth = DepthToPc(depth_img, Range_all)

        # Remove to the origin
        PC_depth = BTL_DataConvert.Npy2Origin(PC_depth)

        # Scale the point cloud
        scale_x = np.max(npy_scan[:, 0]) / np.max(PC_depth[:, 0])
        scale_y = np.max(npy_scan[:, 1]) / np.max(PC_depth[:, 1])
        scale_z = np.max(npy_scan[:, 2]) / np.max(PC_depth[:, 2])

        PC_depth[:, 0] = PC_depth[:, 0] * scale_x
        PC_depth[:, 1] = PC_depth[:, 1] * scale_y
        PC_depth[:, 2] = PC_depth[:, 2] * scale_z

        pcd_scan = open3d.PointCloud()
        pcd_scan.points = open3d.Vector3dVector(PC_depth)
        open3d.draw_geometries([pcd_scan])

    def Powell(self, F, x, obj_A, obj_B, h = 0.1, tol = 1.0e-6):

        # Define a new function
        def f(s):
            return F(obj_A, obj_B, x + s * v)

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
            fOld = F(obj_A, obj_B, x)      # This is not correct -- F(obj_A, obj_B, xOld)

            # First n line searches record decreases of F -- followed by the last line search algorithm
            for i in range(n):

                print("check point")

                # The initial direction on v -- This is im as well
                v = u[i]

                # problem with this cas
                a, b = self.bracket(f, 0.0, h)

                # For the line search only
                s, fMin = self.search(f, a, b, tol=1.0e-9)
                df[i] = fOld - fMin
                fOld = fMin
                x = x + s * v

            # Last line search in the cycle -- this is im -- why this works and how we can prove that?
            v = x - xOld

            # Calculate the bracket
            a, b = self.bracket(f, 0.0, h)
            s, fLast = self.search(f, a, b, tol=1.0e-9)
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

    def bracket(self, f, x1, h):

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

    def Fun(self, x):

        return x[0] * x[0] + 2 * x[1] * x[1] - 4 * x[0] - 2 * x[0] * x[1]

    def search(self, f, a, b, tol = 1.0e-9):

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
        if nIter > 10:
            nIter = 10

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

    # def Optimize_Powell(self):
    #
    #     # Reference:
    #     # https://blog.csdn.net/shenziheng1/article/details/51028074
    #     # https://blog.csdn.net/shenziheng1/article/details/51088738
    #     # https://www.codelast.com/%E5%8E%9F%E5%88%9B-powell%E7%AE%97%E6%B3%95%EF%BC%88powell-algorithm%EF%BC%89%E7%9A%84%E5%AE%9E%E7%8E%B0/
    #     # http://dxyq.tykj.gov.cn/kuai_su/youhuasheji/suanfayuanli/2.2.asp
    #
    #     # Optimize with powell method
    #     Err = 0.01
    #     X0 = np.asarray([[1, 1]])
    #     Dir = np.asarray([[1, 0],
    #                       [0, 1]])
    #     num = 2
    #     # Define the extreme positions and values
    #     X = np.zeros([1, X0.shape[1]])
    #     VAL = np.zeros([1, X0.shape[1]])
    #
    #     # A = np.vstack([A, newrow])
    #     # Step 1 -- Find the optimization results
    #     # Search the max(min) towards the first direction
    #     # print(X0)
    #
    #     [Ext_pos, Ext_val] = self.Optimize_GoldenLineSearch(X0, Dir[0, :], Err)
    #     X[0,:] = Ext_pos
    #     VAL[0,:] = Ext_val
    #
    #     # print(X)
    #     print(VAL)
    #     print(range(1, num - 1))
    #
    #     # for k in range(1, num):
    #     #     # print("k = ", k)
    #     #     dir[]

    # def Optimize_GoldenLineSearch(self, x1, Dir, err):
    #
    #     # Reference: https://blog.csdn.net/shenziheng1/article/details/51088738
    #     # x1: input the value of one direction
    #     # Dir: The search direction
    #     # err: error requirement
    #
    #     # Step 1: Determine the search region
    #     # Determine the input and output values
    #     y1 = self.Fun(x1)    # The function value of the input value
    #     x2 = x1 + Dir        # We need to determine the dimension
    #     # dimen =
    #     y2 = self.Fun(x2)
    #
    #     # Determine the search region
    #     if y1 < y2:
    #         Dir = -Dir
    #         temp = x1
    #         x1 = x2
    #         x2 = temp
    #         x3 = x2 + Dir
    #         y3 = self.Fun(x3)
    #     else:
    #         Dir = 2 * Dir
    #         x3 = x2 + Dir
    #         y3 = self.Fun(x3)
    #
    #     # Find the range
    #     while(True):
    #         if (y2 <= y3):
    #             print(Dir)
    #             print(x1)
    #             print(x3)
    #             a = np.minimum(x1, x3)
    #             b = np.maximum(x1, x3)
    #             break
    #         else:
    #             x1 = x2
    #             x2 = x3
    #             y2 = y3
    #             x3 = x2 + Dir
    #             y3 = self.Fun(x3)
    #
    #     # Golden search
    #     g1 = 0.382
    #     g2 = 0.618
    #     xx1 = a + g1 * (b - a)
    #     yy1 = self.Fun(xx1)
    #     xx2 = a + g2 * (b - a)
    #     yy2 = self.Fun(xx2)
    #
    #     delta = np.sqrt(np.sum(np.fabs(b - a) * np.fabs(b - a)))
    #
    #     delta = np.asarray(delta, dtype = np.float64)
    #     err = np.asarray(err, dtype = np.float64)
    #
    #     print("The delta is ", delta)
    #     print("The err is ", err)
    #     print(delta)
    #     print(err)
    #     # print(delta.all())
    #
    #     while ((delta >= err)) :
    #
    #         # print(1)
    #         print("yy1 = ", yy1)
    #         print("yy2 = ", yy2)
    #         print( yy1 < yy2 )
    #
    #         if (yy1 < yy2):
    #             b = xx2
    #             xx2 = xx1
    #             yy2 = yy1
    #             xx1 = a + g1 * (b - a)
    #             yy1 = self.Fun(xx1)
    #         else:
    #             a = xx1
    #             xx1 = xx2
    #             yy1 = yy2
    #             xx2 = a + g2 * (b - a)
    #             yy2 = self.Fun(xx2)
    #
    #         print(1)
    #
    #         delta = np.sqrt(np.sum(abs(b - a) * abs(b - a)))
    #
    #         print(delta)
    #
    #     # The final result
    #     ExtremePos = 0.5 * (a + b)
    #     ExtremeVal = self.Fun(ExtremePos)
    #
    #     return ExtremePos, ExtremeVal

    def test(self):

        xStart = np.array([1.0, 1.0])
        obj_A = 0
        obj_B = 0
        xMin, nIter = self.Powell(self.Fun, xStart, obj_A, obj_B)

        print("x =", xMin)
        print("F(x) =", self.Fun(xMin))
        print("Number of cycles =", nIter)
        input("Press return to exit")

class PatchGrid():

    def __init__(self):

        # Read the brain pcd file
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()

        # Read the brain and scan npy files
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

        # Define the box variable
        self.obj_globalbox = self.brain_npy
        self.obj_globalbox[:, 0] = self.obj_globalbox[:, 0] * 3
        self.obj_globalbox[:, 1] = self.obj_globalbox[:, 1] * 3
        self.obj_globalbox[:, 2] = self.obj_globalbox[:, 2] * 3

        # Define the box regions
        self.BOX = []
        self.GRID = []

        # Define the joint histogram
        self.Hist = []

        # Define the target object
        self.A = self.brain_npy
        self.B = self.brain_npy
        self.B = self.ApplyTransform(self.B)

    def ApplyTransform(self, obj_npy):

        # Apply transform to the numpy object
        theta_x = 1
        theta_y = 2
        theta_z = 3
        theta = [theta_x, theta_y, theta_z]
        R_tform = BTL_DataConvert.AglTransform(theta[0], theta[1], theta[2])
        obj_npy = np.asarray(np.matmul(obj_npy, R_tform))           # Apply the transform
        obj_npy = BTL_DataConvert.Npy2Origin(obj_npy)               # Move to the origin

        return obj_npy

    def PlotGrid(self, box, obj_npy):

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

    def SetGrid(self, N, obj):

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

    def CAL_Grid(self, Box, obj_npy):

        # Define the GRID object to save the voxel point sets
        GRID = []

        # Plot the grid in different ways
        mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])

        # Get the box within this region
        pcdlist = [mesh_frame]
        pcdlist_nonempty = [mesh_frame]

        for i in range(len(Box)):
            box = Box[i, :]
            pcd = self.PlotGrid(box, obj_npy)
            # print("The current box id is ", box[6])
            # Save only the non-empty region
            pcdlist.append(pcd)
            if len(np.asarray(pcd.points)) != 0:
                pcdlist_nonempty.append(pcd)

        # Save only the grid regions
        GRID = pcdlist[1:]
        # print("The length of non empty girds are ", len(pcdlist_nonempty))

        return GRID

    def Histogram(self, GRID):

        # Show the intensity with the voxel ID
        Hist = np.random.normal(size = len(GRID))
        for idx, item_box in enumerate(GRID):
            # print(len(np.asarray(item_box.points)))
            # Solve the zero grid problem
            # if(np.int(len(np.asarray(item_box.points))) == 729):
            #     print("The check idx is ", idx)
            # else:
            Hist[idx] = np.int(len(np.asarray(item_box.points)))

        # Calculate the joint histogram
        # plt.hist(Hist, bins = len(Hist))
        # plt.show()
        # print(np.max(Hist))

        return Hist

    def JointHist(self, Hist_A, Hist_B):

        # Read the one dimensional histogram
        fig, axes = plt.subplots(1, 2)

        # The hist for object A
        hist_A = np.histogram(Hist_A, bins = len(Hist_A))
        # print("The histogram of A is ", len(hist_A[0]))
        # axes[0].hist(Hist_A, bins = 20)
        # axes[0].set_title('T1 slice histogram')

        # The hist for object B
        hist_B = np.histogram(Hist_B, bins = len(Hist_B))
        # axes[1].hist(Hist_B, bins = 20)
        # axes[1].set_title('T2 slice histogram')
        # plt.show()

        # Joint histogram -- 1D
        # print("The length of histogram A is ", Hist_A)
        # print(len(Hist_A.ravel()))
        # plt.plot(Hist_A.ravel(), Hist_B.ravel(), '.')
        # plt.plot(hist_A[0][1:], hist_B[0][1:], '.')
        # plt.show()

        # Joint histogram -- 2D
        hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
        # print(hist_2d.T.shape)
        hist_2d.T[0][0] = 0
        # plt.imshow(hist_2d.T, origin='lower')
        # plt.show()

        return hist_2d

    def PlotPcdlist(self, npy_list):

        mesh_frame = open3d.create_mesh_coordinate_frame(size=0.6, origin=[0, 0, 0])
        PCD = [mesh_frame]

        # Plot the pcd object
        for item in npy_list:
            pcd = open3d.PointCloud()
            pcd.points = open3d.Vector3dVector(item)
            PCD.append(pcd)

        open3d.draw_geometries(PCD)

    def Test(self):

        # Get the box for object A, B
        N_A = 10
        N_B = 10
        Box_A = self.SetGrid(N_A, self.obj_globalbox)
        Box_B = self.SetGrid(N_B, self.obj_globalbox)

        # Test the CAL_Grid
        Grid_A = self.CAL_Grid(Box_A, self.A)
        Grid_B = self.CAL_Grid(Box_B, self.B)

        # Test the histogram function
        Hist_A = self.Histogram(Grid_A)
        Hist_B = self.Histogram(Grid_B)

        # The joint histogram -- 1D
        self.JointHist(Hist_A, Hist_B)

        # Show the two pcd objects
        self.PlotPcdlist([self.A, self.B])

def DepthToPc(depth_img, Range_all):

    # Depth image to point cloud
    a = 1
    h = depth_img.shape[0]
    w = depth_img.shape[1]
    PC_depth = np.zeros((h * w, 3))

    row = 0
    for i in range(h):
        for j in range(w):
            z = depth_img[i][j]
            print(z)
            PC_depth[row, 0] = i
            PC_depth[row, 1] = j
            PC_depth[row, 2] = z
            row += 1
    print(PC_depth)
    print(PC_depth.shape)
    pcd_scan = open3d.PointCloud()
    pcd_scan.points = open3d.Vector3dVector(PC_depth)
    open3d.draw_geometries([pcd_scan])

    return PC_depth

if __name__ == "__main__":

    # test = Register()
    # test.Test()

    test = MIR()
    test.test()

    # test.DataUpsample()
    # test.Optimize_Powell()
