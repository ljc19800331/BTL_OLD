# Goals:
# 1. Voxelize the space
# 2. Order the space
# 3. Get the 3D features
# 4. Plot the histogram -- if possible

import numpy as np
# from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import BTL_Registration
import open3d
import matplotlib.pyplot as plt
import BTL_DataConvert

# Algorithm:
# 1. PlotGrid:

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
        theta_y = 1
        theta_z = 0.5
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
        print("The length of non empty girds are ", len(pcdlist_nonempty))

        return GRID

    def Histogram(self, GRID):

        # Show the intensity with the voxel ID
        Hist = np.random.normal(size = len(GRID))
        for idx, item_box in enumerate(GRID):
            print(len(np.asarray(item_box.points)))
            # Solve the zero grid problem
            # if(np.int(len(np.asarray(item_box.points))) == 729):
            #     print("The check idx is ", idx)
            # else:
            Hist[idx] = np.int(len(np.asarray(item_box.points)))

        # Calculate the joint histogram
        plt.hist(Hist, bins = len(Hist))
        plt.show()
        print(np.max(Hist))

        return Hist

    def JointHist(self, Hist_A, Hist_B):

        # Read the one dimensional histogram
        fig, axes = plt.subplots(1, 2)

        # The hist for object A
        hist_A = np.histogram(Hist_A, bins = len(Hist_A))
        # print("The histogram of A is ", len(hist_A[0]))
        axes[0].hist(Hist_A, bins = 20)
        axes[0].set_title('T1 slice histogram')

        # The hist for object B
        hist_B = np.histogram(Hist_B, bins = len(Hist_B))
        axes[1].hist(Hist_B, bins = 20)
        axes[1].set_title('T2 slice histogram')
        plt.show()

        # Joint histogram -- 1D
        # print("The length of histogram A is ", Hist_A)
        # print(len(Hist_A.ravel()))
        plt.plot(Hist_A.ravel(), Hist_B.ravel(), '.')
        # plt.plot(hist_A[0][1:], hist_B[0][1:], '.')
        plt.show()

        # Joint histogram -- 2D
        hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
        print(hist_2d.T.shape)
        hist_2d.T[0][0] = 0
        plt.imshow(hist_2d.T, origin='lower')
        plt.show()

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

if __name__ == "__main__":

    test = PatchGrid()
    test.Test()
    # test.Histogram()
    # box = np.asarray([0, 30, 0, 30, 0, 30])
    # test.PlotGrid(box)
    # test.CAL_Grid()
    # test.CubePlot()