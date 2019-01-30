# This is used for textured mapping based colorized point cloud generation
'''
1. plane colorized point cloud
    Create a plane point cloud
    divide the region into several parts
2. non-plane colorized point cloud -- hemisphere model
3. more complicated model if possible
4. Brain-based colorized point cloud

Code structure:
1. function -- textured image
2. function -- show the point cloud
3. function -- ImgMapPc

Reference:
1. Textured mapping in Python: https://plot.ly/~empet/14172/mapping-an-image-on-a-surface/#/

'''

import cv2
import open3d
import vtk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class BTL_GL():

    def __init__(self):
        self.img_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_cortical.jpg"
        self.pc_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain.ply"

    def GL_NonFlat(self):

        # Non flat plane
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape
        img = np.fliplr(img[0:w - 1, 0:h - 1])
        img = np.fliplr(img[0:w - 1, 0:h - 1])
        print(img)
        pc = 1

        # Read the point cloud
        pc, xyz = self.TexPc()
        points = xyz[:, 0:2]
        values = xyz[:, 2]
        print(points.shape)
        print(values.shape)

        x_max = np.max(xyz[:, 0])
        y_max = np.max(xyz[:, 1])
        grid_x, grid_y = np.mgrid[0:x_max:complex(w - 1), 0:y_max:complex(h - 1)]

        grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

        pc = np.zeros((len(grid_x.flatten()), 3))
        pc[:, 0] = grid_x.flatten()
        pc[:, 1] = grid_y.flatten()
        pc[:, 2] = grid_z.flatten()

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc)

        # Assign the color image
        [w, h, d] = img.shape
        print("The width of the image is ", w)
        print("The height of the image is ", h)
        vec_color = img.reshape(w * h, 3)
        print(vec_color / 255)
        # print(vec_color[:,0])
        vec_color = open3d.Vector3dVector(np.asarray(vec_color / 255))
        pcd.colors = vec_color
        open3d.draw_geometries([pcd])

    def GL_Flat(self):

        # Create the flat plane of the object
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape
        img = np.fliplr(img[0:w-1, 0:h-1])
        print(img)
        pc = 1
        x_max = 199
        y_max = 199
        grid_x, grid_y = np.mgrid[0:x_max:complex(w-1), 0:y_max:complex(h-1)]
        print("grid_x is ", grid_x)
        print("grid_y is ", grid_y)
        grid_z = np.ones((len(grid_x.flatten()),1))
        pc = np.zeros((len(grid_x.flatten()), 3))
        pc[:,0] = grid_x.flatten()
        pc[:,1] = grid_y.flatten()
        pc[:,2] = grid_z.flatten()
        print(pc)
        print(pc.shape)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc)
        # open3d.draw_geometries([pcd])

        # Assign the color image
        [w, h, d] = img.shape
        print("The width of the image is ", w)
        print("The height of the image is ", h)
        vec_color = img.reshape(w * h, 3)
        print(vec_color/255)
        # print(vec_color[:,0])
        vec_color = open3d.Vector3dVector(np.asarray(vec_color / 255))
        pcd.colors = vec_color
        open3d.draw_geometries([pcd])

    def TexImg(self):
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)    # Notice: BGR to RGB image
        plt.imshow(img)
        plt.show()

    def TexPc(self):
        pc = open3d.read_point_cloud("/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain.ply")
        xyz = np.asarray(pc.points)
        # open3d.draw_geometries([pc])
        return pc, xyz

    def TexMap(self):
        # Map the image to the region
        a = 1

if __name__ == "__main__":

    test = BTL_GL()
    test.GL_NonFlat()
    # test.GL_Flat()
    # test.TexImg()
    # test.TexPc()