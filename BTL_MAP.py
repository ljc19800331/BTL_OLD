import numpy as np
import BTL_VIZ
import cv2
from BTL_VIZ import *
from plyfile import PlyData
from skimage.feature import match_template
from scipy.interpolate import griddata
import DataConvert
from DataConvert import *
from matplotlib import pyplot as plt
from scipy import ndimage
import time

class AutoPcMap:

    def __init__(self):
        self.scan = []
        self.brain = []
        self.model_x = 3.346  # inch
        self.model_y = 4.547  # inch
        self.model_z = 1.571  # inch
        self.res = 0.5        # The size of the scan box
        self.pixel_res = 100  # Change during the time
        self.step = 100       # inch

    def txt2pt(self, txtscan_x, txtscan_y, txtscan_z):

        # Initialize
        a_x = []; a_y = []; a_z = []
        scan_x = []; scan_y = []; scan_z = []

        # Load the data
        scan_L1x = open(txtscan_x, "r")
        for line in scan_L1x:
            a_x = line.split()
            scan_x.append(float(a_x[0]))

        scan_L1y = open(txtscan_y, "r")
        for line in scan_L1y:
            a_y = line.split()
            scan_y.append(float(a_y[0]))

        scan_L1z = open(txtscan_z, "r")
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

        # Save to npy
        npy_scan = np.zeros((len(scan_x), 3), dtype=np.float32)
        npy_scan[:,0] = scan_x
        npy_scan[:,1] = scan_y
        npy_scan[:,2] = scan_z

        # convert to the point cloud -- vtk format
        vtk_scan = BTL_VIZ.VtkPointCloud()
        for k in range(len(scan_x)):
            point = ([scan_x[k], scan_y[k], scan_z[k]])
            vtk_scan.addPoint(point)

        return vtk_scan, npy_scan, scan_x, scan_y, scan_z

    def Brain_Init(self, txtbrain_x, txtbrain_y, txtbrain_z, theta):

        # Load the data
        vtk_brain, npy_brain, brain_x, brain_y, brain_z = self.txt2pt(txtbrain_x, txtbrain_y, txtbrain_z)

        # Define the transformation matrix
        theta_x = theta[0]; theta_y = theta[1]; theta_z = theta[2]
        R_tform = DataConvert.AglTransform(theta_x, theta_y, theta_z)
        pc_tform = np.matmul(npy_brain, R_tform)
        pc_tform = np.asarray(pc_tform)             # The difference between ndarray and asanyarray
        pc_tform = DataConvert.pt2Len(pc_tform, 'brain', 'inch')

        # Change the size parameter to inch
        pc_tform = DataConvert.Npy2Origin(pc_tform)

        # Change the format to the npy data
        vtk_brain = DataConvert.npy2vtk(pc_tform)

        return vtk_brain, pc_tform, pc_tform[:,0], pc_tform[:,1], pc_tform[:,2]

    def Scan2img(self, npy_scan, nx, ny):

        # Define the transform angle
        theta = [45, 45, 0]

        # Apply the transformation
        R_tform = DataConvert.AglTransform(theta[0], theta[1], theta[2])
        npy_obj = np.asarray(np.matmul(npy_scan, R_tform))
        npy_obj = DataConvert.Npy2Origin(npy_obj)

        # This function aims to convert the 3D scan point cloud to the image
        x_obj = npy_obj[:,0]
        y_obj = npy_obj[:,1]
        z_obj = npy_obj[:,2]

        # The resolution of the grid
        x_grid = np.linspace(np.min(x_obj), np.max(x_obj), nx)
        y_grid = np.linspace(np.min(y_obj), np.max(y_obj), ny)
        xv, yv = np.meshgrid(x_grid, y_grid)

        # Interpolate the grid based on the
        grid_x, grid_y = np.mgrid[np.min(x_obj) : np.max(x_obj) : np.max(x_obj) / nx, \
                                  np.min(y_obj) : np.max(y_obj) : np.max(y_obj) / ny]
        points = npy_obj[:, 0:2]
        values = npy_obj[:, 2]
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        rotated = ndimage.rotate(grid_z0, 90)
        frame = np.asarray(cv2.flip(rotated, 1))
        # plt.subplot(221), plt.imshow(frame, 'gray'), plt.title('ORIGINAL')
        # plt.show()

        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        return npy_obj, frame

    def Brain2Img(self, npy_brain):

        # Data normalization
        brain_x = npy_brain[:,0]
        brain_y = npy_brain[:,1]
        brain_z = npy_brain[:,2]
        brain_xarray = npy_brain[:, 0]
        brain_yarray = npy_brain[:, 1]
        points_brain = np.column_stack((brain_xarray, brain_yarray))
        values_brain = npy_brain[:, 2]

        # Get the range of x and y
        x_max = np.max(brain_x)
        x_min = np.min(brain_x)
        y_max = np.max(brain_y)
        y_min = np.min(brain_y)
        z_max = np.max(brain_z)
        z_min = np.min(brain_z)

        # Define the self resolution
        N_step = self.step

        # Determine the number of pixels in x and y axis
        N_x = round( np.max(npy_brain[:,0]) / self.res * N_step)
        N_y = round( np.max(npy_brain[:,1]) / self.res * N_step)

        # Create the mesh grid mapping
        grid_x, grid_y = np.mgrid[0 :x_max :x_max / N_x, 0 :y_max :y_max / N_y]
        grid_z0 = -griddata(points_brain, values_brain, (grid_x, grid_y), method='nearest')
        grid_ranges = [[0, x_max, x_max / N_x], [0, y_max, y_max / N_y], [0, x_max, x_max / N_x]]
        img_brain = np.rot90(grid_z0)

        return img_brain

    def ScanMapBrain(self, img_brain, img_scan):

        # This function aims to register the image to the global framework
        # output: return a registration coefficient
        # Match the result using normalize cross correlation
        match_result = match_template(img_brain, img_scan)

        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        ij = np.unravel_index(np.argmax(match_result), match_result.shape)
        x, y = ij[::-1]

        x_center = x + self.pixel_res/2
        y_center = y + self.pixel_res/2

        flag_type = 'scan'
        p_center = [x_center, y_center]
        model_Len = [self.model_x, self.model_y]
        [Len_x, Len_y] = pixel2Len(p_center, img_brain, flag_type, model_Len)

        # viz the result on the image
        top_left = (x, y)
        bottom_right = (x_center + self.pixel_res/2, y_center + self.pixel_res/2)
        cv2.rectangle(img_brain, top_left, bottom_right, (255,0,0), 15)
        cv2.imshow('image', img_brain)
        cv2.waitKey(0)

        return [Len_x, Len_y], img_brain, img_scan

    # ICP to perform fine registration
    # Location of the laser spot (surgical tool) in the pMR image

if __name__ == "__main__":

    # Test txt2pt
    # test = AutoPcMap()
    # txtscan_x = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_x'
    # txtscan_y = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_y'
    # txtscan_z = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_z'
    # vtk_scan, npy_scan, scan_x, scan_y, scan_z = test.txt2pt(txtscan_x, txtscan_y, txtscan_z)
    # vtk_data = DataConvert.npy2vtk(npy_scan)
    # VizVtk([vtk_data])

    # Test brain_init
    # test = AutoPcMap()
    # txtbrain_x = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_x.txt'
    # txtbrain_y = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_y.txt'
    # txtbrain_z = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_z.txt'
    # theta = [0, 0, 180]
    # vtk_brain, pc_tform, pc_tform_x, pc_tform_y, pc_tform_z = test.Brain_Init(txtbrain_x, txtbrain_y, txtbrain_z, theta)

    # Project the point cloud data to the plane
    test = AutoPcMap()
    txtscan_x = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_x'
    txtscan_y = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_y'
    txtscan_z = '/home/maguangshen/PycharmProjects/BTL_GS/Data/ScanData(NonRotate)/9/Scan_z'

    vtk_scan, npy_scan, scan_x, scan_y, scan_z = test.txt2pt(txtscan_x, txtscan_y, txtscan_z)
    nx = 100; ny = 100
    npy_obj, frame = test.Scan2img(npy_scan, nx, ny)

    # vtk_data = DataConvert.npy2vtk(npy_obj)
    # VizVtk([vtk_data])

    img_brain = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/BrainImg/Source images/img_3D_box5_dis5_iter5_xy/0_0.png', 0)
    img_scan = frame
    # print(type(img_brain))
    # print(type(img_scan))

    [Len_x, Len_y], img_brain, img_scan = test.ScanMapBrain(img_brain, img_scan)

