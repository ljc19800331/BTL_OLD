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
# 0.
# 1. Data preprocessing module -- define X and Y grid together
#       1.1 init function
#       1.2 DataUpsamling function
#       1.3 Find an easy case
# 2. Optimization module -- powell method -- pending
#       2.1 Powell method
# 3. Cost function definition module
# 4. Mutual information module
# 5. Registration process module
import open3d
from open3d import *
import numpy as np
import BTL_DataConvert
import BTL_MAP
import math

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

    def DataInterpolate(self):
        # Interpolate the 3D Point grid
        a = 1

    def Powell(self, F, x, h = 0.1, tol = 1.0e-6):

        # Define a new function
        def f(s):
            return F(x + s * v)

        n = len(x)  # number of design variables
        df = np.zeros(n)  # Decreases of F stored here
        u = np.identity(n)  # Initial vectors here by rows

        for j in range(30):

            # Allow for only 30 cycles (loops) - maximum n times -- no less then 20 is good enough
            # j is the number of iteration -- this is a good idea

            xOld = x.copy()  # The input x is usually the xStart point
            fOld = F(xOld)

            # First n line searches record decreases of F -- followed by the last line search algorithm
            for i in range(n):
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

            # Problem with this sentence
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
            h = c * h
            x3 = x2 + h
            f3 = f(x3)
            if f3 > f2:
                return x1, x3
            x1 = x2
            x2 = x3
            f1 = f2
            f2 = f3

    def search(self, f, a, b, tol = 1.0e-9):

        # Initial position
        # Calculate the total iteration numbers
        nIter = int(math.ceil(-2.078087 * math.log(tol / abs(b - a))))
        R = 0.618033989
        C = 1.0 - R

        # First telescoping -- Define the golden length
        x1 = R * a + C * b
        x2 = C * a + R * b
        f1 = f(x1)
        f2 = f(x2)

        # Main loop
        for i in range(nIter):

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

    def Optimize_Powell(self):

        # Reference:
        # https://blog.csdn.net/shenziheng1/article/details/51028074
        # https://blog.csdn.net/shenziheng1/article/details/51088738
        # https://www.codelast.com/%E5%8E%9F%E5%88%9B-powell%E7%AE%97%E6%B3%95%EF%BC%88powell-algorithm%EF%BC%89%E7%9A%84%E5%AE%9E%E7%8E%B0/
        # http://dxyq.tykj.gov.cn/kuai_su/youhuasheji/suanfayuanli/2.2.asp

        # Optimize with powell method
        Err = 0.01
        X0 = np.asarray([[1, 1]])
        Dir = np.asarray([[1, 0],
                          [0, 1]])
        num = 2
        # Define the extreme positions and values
        X = np.zeros([1, X0.shape[1]])
        VAL = np.zeros([1, X0.shape[1]])

        # A = np.vstack([A, newrow])
        # Step 1 -- Find the optimization results
        # Search the max(min) towards the first direction
        # print(X0)

        [Ext_pos, Ext_val] = self.Optimize_GoldenLineSearch(X0, Dir[0, :], Err)
        X[0,:] = Ext_pos
        VAL[0,:] = Ext_val

        # print(X)
        print(VAL)
        print(range(1, num - 1))

        # for k in range(1, num):
        #     # print("k = ", k)
        #     dir[]

    def Optimize_GoldenLineSearch(self, x1, Dir, err):

        # Reference: https://blog.csdn.net/shenziheng1/article/details/51088738
        # x1: input the value of one direction
        # Dir: The search direction
        # err: error requirement

        # Step 1: Determine the search region
        # Determine the input and output values
        y1 = self.Fun(x1)    # The function value of the input value
        x2 = x1 + Dir        # We need to determine the dimension
        # dimen =
        y2 = self.Fun(x2)

        # Determine the search region
        if y1 < y2:
            Dir = -Dir
            temp = x1
            x1 = x2
            x2 = temp
            x3 = x2 + Dir
            y3 = self.Fun(x3)
        else:
            Dir = 2 * Dir
            x3 = x2 + Dir
            y3 = self.Fun(x3)

        # Find the range
        while(True):
            if (y2 <= y3):
                print(Dir)
                print(x1)
                print(x3)
                a = np.minimum(x1, x3)
                b = np.maximum(x1, x3)
                break
            else:
                x1 = x2
                x2 = x3
                y2 = y3
                x3 = x2 + Dir
                y3 = self.Fun(x3)

        # Golden search
        g1 = 0.382
        g2 = 0.618
        xx1 = a + g1 * (b - a)
        yy1 = self.Fun(xx1)
        xx2 = a + g2 * (b - a)
        yy2 = self.Fun(xx2)

        delta = np.sqrt(np.sum(np.fabs(b - a) * np.fabs(b - a)))

        delta = np.asarray(delta, dtype = np.float64)
        err = np.asarray(err, dtype = np.float64)

        print("The delta is ", delta)
        print("The err is ", err)
        print(delta)
        print(err)
        # print(delta.all())

        while ((delta >= err)) :

            # print(1)
            print("yy1 = ", yy1)
            print("yy2 = ", yy2)
            print( yy1 < yy2 )

            if (yy1 < yy2):
                b = xx2
                xx2 = xx1
                yy2 = yy1
                xx1 = a + g1 * (b - a)
                yy1 = self.Fun(xx1)
            else:
                a = xx1
                xx1 = xx2
                yy1 = yy2
                xx2 = a + g2 * (b - a)
                yy2 = self.Fun(xx2)

            print(1)

            delta = np.sqrt(np.sum(abs(b - a) * abs(b - a)))

            print(delta)

        # The final result
        ExtremePos = 0.5 * (a + b)
        ExtremeVal = self.Fun(ExtremePos)

        return ExtremePos, ExtremeVal

    def Fun(self, x):

        return x[0] * x[0] + 2 * x[1] * x[1] - 4 * x[0] - 2 * x[0] * x[1]

    def test(self):

        xStart = np.array([1.0, 1.0])
        xMin, nIter = self.Powell(self.Fun, xStart)

        print("x =", xMin)
        print("F(x) =", self.Fun(xMin))
        print("Number of cycles =", nIter)
        input("Press return to exit")

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

    test = MIR()
    test.test()

    # test.DataUpsample()
    # test.Optimize_Powell()
