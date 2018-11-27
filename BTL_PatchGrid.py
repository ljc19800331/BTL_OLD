import numpy as np
# from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import BTL_Registration
import open3d

class PatchGrid():

    def __init__(self):
        # Test the results
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()

        # Open3d([brain_pcd, scan_pcd])
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

        # Define the box variable
        self.BOX = []

    def PlotGrid(self, box):

        # Plot the grid inside the cube region
        a = 1

        # box = np.asarray([0, 30, 0, 50, 0, 50])

        # print(box)
        model = self.brain_npy
        idx = np.where( (model[:, 0] > box[0]) & (model[:, 0] < box[1])  \
                        & (model[:, 1] > box[2]) & (model[:, 1] < box[3]) \
                        & (model[:, 2] > box[4]) & (model[:, 2] < box[5]) )
        obj = model[idx, :]

        # print(idx[0])
        obj = obj.reshape((len(idx[0]), 3))

        # print(obj.shape)

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(obj)
        # open3d.draw_geometries([pcd])

        return pcd

    def SetGrid(self, N):

        # The initial definiton of our parameter setting
        x_min = np.min(self.brain_npy[:, 0])
        x_max = np.max(self.brain_npy[:, 0])
        y_min = np.min(self.brain_npy[:, 1])
        y_max = np.max(self.brain_npy[:, 1])
        z_min = np.min(self.brain_npy[:, 2])
        z_max = np.max(self.brain_npy[:, 2])

        # 3D meshgrid
        # N = 11
        x = np.linspace(x_min, x_max, N)
        y = np.linspace(y_min, y_max, N)
        z = np.linspace(z_min, z_max, N)

        # print(x)
        # print(y)
        # print(z)
        # X, Y, Z = np.mgrid[-1:1:11, -1:1:11, -1:1:11]
        # X, Y, Z = np.mgrid[x_min:x_max:N, y_min:y_max:N, z_min:z_max:N]
        # X, Y, Z = np.mgrid[x, y]

        # 3D box Creation
        Dict = np.zeros([(len(x) - 1) * (len(y)-1) * (len(z)-1), 6])
        count = 0
        for idx_x in range(len(x) - 1):
            x1 = x[idx_x]
            x2 = x[idx_x + 1]
            for idx_y in range(len(y) - 1):
                y1 = y[idx_y]
                y2 = y[idx_y + 1]
                for idx_z in range(len(z) - 1):
                    print(count)
                    z1 = z[idx_z]
                    z2 = z[idx_z + 1]
                    IDX = np.asarray([idx_x, idx_y, idx_z])
                    # LIST = [x1, x2, y1, y2, z1, z2]
                    Dict[count, :] = np.asarray([x1, x2, y1, y2, z1, z2])
                    count += 1
        self.BOX = Dict
        print(Dict)

    def CubePlot(self):

        # prepare some coordinates
        x, y, z = np.indices((10, 10, 10))

        # Draw cuboids in the top left and bottom right corners, and a link between them
        cube1 = (x < 3) & (y < 3) & (z < 3)
        cube2 = (x >= 5) & (y >= 5) & (z >= 5)
        link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

        # Combine the objects into a single boolean array
        voxels = cube1 | cube2 | link

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[link] = 'red'
        colors[cube1] = 'blue'
        colors[cube2] = 'green'

        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')

        plt.show()

    def TEST_Grid(self):

        # Plot the grid in different ways
        mesh_frame = open3d.create_mesh_coordinate_frame(size=0.6, origin=[0, 0, 0])
        self.SetGrid(10)

        # Get the box
        pcdlist = [mesh_frame]
        for i in range(len(self.BOX)):
            print(i)
            box = self.BOX[i,:]
            # print(pcd.points)
            pcd = self.PlotGrid(box)
            print(np.asarray(pcd.points))
            if len(np.asarray(pcd.points)) != 0:
                pcdlist.append(pcd)

        print(pcdlist[1])
        obj_plot = []
        for i in range(len(pcdlist)):
            obj_plot.append(pcdlist[i])

        open3d.draw_geometries([pcdlist[100], pcdlist[200], pcdlist[300]])

if __name__ == "__main__":
    test = PatchGrid()
    test.TEST_Grid()