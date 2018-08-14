# Calibration PNP
# ref: # https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
# https://github.com/IntelRealSense/librealsense/issues/869
# # https://www.youtube.com/watch?v=NtJXi7u_fJo
from angleConvert import AngleConvert
from controlDAQ import ControlDAQ
from controlMovement import ControlMovement
from convertSteering import ConvertSteering
from generateScanAngles import GenerateScanAngles
from scanTypes_GS import *
import TwoImageRegistration
from TwoImageRegistration import *
from BTL_SEG import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyrealsense2 as rs
from threading import Thread
import time
import os, os.path
import vtk
from DataConvert import *
import BTL_MultiProcess
import thread
import BTL_VIZ
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Tkinter import *
from tkFileDialog import askopenfilename
import PIL
from PIL import Image
from PIL import ImageTk

class PNP:

    def __init__(self):

        self.AC = AngleConvert()
        self.CM = ControlMovement()
        self.ST = ScanTypes()
        self.CS = ConvertSteering()
        self.CD = ControlDAQ()
        self.GSA = GenerateScanAngles()

        # Design the interval of the time step
        self.arrowMove = 0.05

        # Camera setting
        # self.cap = cv2.VideoCapture(1)

        # Realsense setting
        # flag = raw_input("if viz data? (yes or no)")
        # if flag == 'no':
        #     self.pipeline = rs.pipeline()
        #     self.config = rs.config()
        #     self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #     self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #
        #     # Start streaming
        #     self.pipeline.start(self.config)

    def LaserSpotDetect(self):

        # Define the ROI manually
        img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        print "Please choose the ROI"
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The rectangular region is ", r

        # Detect the region
        # realtime -- realtime image detection
        # static -- static test image detection
        flag = raw_input("realtime or static?")

        if flag == "realtime":
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(config)
                while True:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                    grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
                    Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
                    cv2.circle(color_image, Spot_center, 30, (0,0,0), 2)
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', color_image)
                    cv2.waitKey(1)
            finally:
                # Stop streaming
                pipeline.stop()

        if flag == "static":
            color_image = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
            ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
            Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
            cv2.circle(color_image, Spot_center, 30, (0, 0, 0), 2)
            print "The Laser spot center is ", Spot_center
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(0)

    def CamIntrinsic(self):

        # intrinsic parameters of the depth and color imagers
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # Intrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            print(color_intrin)
        finally:
            # Stop streaming
            pipeline.stop()
            print "Return the color frame intrinsic parameters"

        return color_intrin

    def RsRealtime(self):

        try:
            print "Begin the realtime streaming with the RealSense Camera"
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                cv2.waitKey(1)
        finally:
            # Stop streaming
            pipeline.stop()

    def CapRealtime(self):

        try:
            print "The realtime of the webcam camera"
            cap = cv2.VideoCapture(1)
            while(True):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def MoveOrigin(self):

        print "Move the laser spot to the origin (x_angle = 0, y_angle = 0)"
        x = 0
        y = 0
        self.CD.point_move(x, y, 0.001)

    def MovePoint(self, move_x, move_y):

        print "move a point to a specific position (x,y)"
        print "input move_x as unit(inches)"
        print "input move_y as unit(inches)"

        # update the original .npy files
        filename = 'ScanAngles.npy'
        if os.path.exists(filename):
            os.remove(filename)

        print 'Set center of scan by setting the x and y angles'
        x_c = 0
        y_c = 0

        unit_move = 0.05
        UserIn = Inch2Direction(unit_move, move_x, move_y)
        centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn)

    def ScanPoints(self):

        filename = 'ScanAngles.npy'
        if os.path.exists(filename):
            os.remove(filename)

        mtiScanSpan = [1.8, 1.8]        # The scanning region of the MTI
        pointDistance = 0.3             # inches between each point on the scan

        # Move the center with x_angle == 0 and y_angle == 0
        print 'Set center of scan.'
        x_c = 0; y_c = 0                # The starting angles value -- origin

        # Define the initial position from the origin -- different from (0,0)
        unit_move = 0.05
        UserIn = Inch2Direction(unit_move, 0.96, -0.96)

        flag = raw_input("Do you want to move to the predefined center (yes/no)?")

        if flag == 'yes':
            centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn)
        if flag == 'no':
            centerScan, z_dist, center_angles = self.CM.controlPoint([x_c], [y_c])
        print "The center angles are", center_angles

        print 'Set scan box location and size.'
        centerScan, scanSpread, z_dist = self.CM.controlScanBox(centerScan, z_dist, mtiScanSpan, 'box', 'vertical', '2')
        scanCenter = centerScan

        # z_dist2 = z_dist
        pointsPerX = scanSpread[0] / pointDistance
        pointsPerY = scanSpread[1] / pointDistance

        # print 'Performing initial raster scan.'
        print "The z_dist is ", z_dist
        target_points, filtered_points = self.ST.raster_scan_v1(centerScan, scanSpread, z_dist, pointsPerX, pointsPerY, 'Scan', 'ScanAngles')
        self.ST.plot_scan(target_points, 1)
        self.PointSTA(target_points)

        flag_save = raw_input("Do you want to save the scanning data (yes/no)?")
        if flag_save == 'yes':
            np.save('C:/Users/gm143/TumorCNC_brainlab/BTL/P_MTI_3d_Calibration.npy', target_points)

    def PointSTA(self, Points):

        # Statistical for the points
        # Points = np.random.randint(5, size=(10, 3))

        print("The points are ", Points)
        # min and max z
        z_min = np.min(Points[:, 2])
        z_max = np.max(Points[:, 2])
        diff_zminmax = z_max - z_min
        print "The different of z min and max is ", diff_zminmax

        # mean interval for x
        x = Points[:,0]
        x_1 = x[:-1]
        x_2 = x[1:]
        x_int = np.absolute(x_2 - x_1)
        x_int_mean = np.mean(x_int)
        print "The mean interval of x is ", x_int_mean

        # mean interval for y
        y = Points[:,1]
        y_1 = y[:-1]
        y_2 = y[1:]
        y_int = np.absolute(y_2 - y_1)
        y_int_mean = np.mean(y_int)
        print "The mean interval of y is ", y_int_mean

    def LaserSpotCalibration2D(self):

        # Extract the 2D coordinates from the images
        path, dirs, files = next(os.walk("C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser"))
        img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser/P_1.png')
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The region is ", r

        P_2D = np.zeros([  len(range(1,len(files)+1)), 2 ])
        idx = 0
        for i in range(1, len(files) + 1):
            img_name = 'C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser/' + 'P_' + str(i) + '.png'
            img = cv2.imread(img_name)
            print "The current image is ", img
            ROI_image = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
            Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
            P_2D[idx,:] = Spot_center
            idx += 1

        print "The 2D coordinates are ", P_2D
        print "The shape of the 2D coordinates are ", P_2D.shape

        flag_save = raw_input("Save the 2D coordinates for all the points (yes/no)?")
        if flag_save == 'yes':
            np.save('C:\Users\gm143\TumorCNC_brainlab\BTL\P_2D.npy', P_2D)

    def CaptureMaskImg(self):

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipe_profile = pipeline.start(config)
        curr_frame = 0
        flag_save = raw_input("Do you want to save the image?")

        try:
            while (curr_frame < 20):

                # Get the color frame
                frames = pipeline.wait_for_frames()

                # Align_depth_frame is a 640x480 depth image
                align_to = rs.stream.color
                align = rs.align(align_to)
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                # This is important since the first frame of the color image is dark while later it is better
                curr_frame += 1
                if (curr_frame == 10) and flag_save == 'yes':
                    cv2.imwrite('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png', color_image)
        finally:
            pipeline.stop()

        return color_image

    def Scan3d(self, MOVE_X, MOVE_Y):

        print "Move the MTI spot to specific positions"

        # Initialization
        centerScan = [0, 0]
        x_angle = [0]
        y_angle = [0]
        x_cent_ang = x_angle  # np.array( -1.0381408)
        y_cent_ang = y_angle  # np.array(-1.168893)
        self.CD.point_move(x_cent_ang, y_cent_ang, .005)
        z_dist = float(self.CD.point_measure(1))
        z_dist = float((z_dist / 0.04 + 175) * 0.03937007874)
        org_targetPoints = self.CS.xy_position(x_cent_ang, y_cent_ang, z_dist)
        centerScan[0] = org_targetPoints[0, 0]
        centerScan[1] = org_targetPoints[0, 1]
        z_dist = org_targetPoints[0, 2]
        print "The centerScan is ", centerScan
        print "The z_dist is ", z_dist

        arrowMove = 0.05            # This is self-define
        P_MTI = np.zeros([len(MOVE_X), 3])

        for i in range(len(MOVE_X)):

            move_x = MOVE_X[i]             # This is self-define
            move_y = MOVE_Y[i]

            centerScan[0] = org_targetPoints[0, 0] + move_x
            centerScan[1] = org_targetPoints[0, 1] + move_y

            # print('The centerScan is', centerScan)
            [x_cent_ang, y_cent_ang] = self.CS.angular_position(np.array([centerScan[0]]),
                                                                np.array([centerScan[1]]),
                                                                np.array([z_dist]))             # recalculate the center
            self.CD.point_move(x_cent_ang, y_cent_ang, 0.005)                                   # move to the new point
            z_dist = float(self.CD.point_measure(0.1))                                          # measure a new distance (this returns a voltage)
            z_dist2 = (z_dist / 0.04 + 175) * 0.03937007874                                     # convert the voltage to a distance
            targetPoints = self.CS.xy_position(x_cent_ang, y_cent_ang, z_dist2)                 # figure out the actual point in space we are at
            z_dist = targetPoints[0, 2]                                                         # return that distance
            print "The current target points are ", z_dist
            P_MTI[i, :] = np.asarray([move_x, move_y, z_dist])

        print "The points of MTI are ", P_MTI
        flag_save = raw_input("Do you want to save the data?")
        if flag_save == 'yes':
            np.save('P_MIT.npy', P_MTI)

        # MOVE_X = -np.linspace(0.0, 1.8, num=7)
        # MOVE_Y = -np.linspace(0.0, 1.8, num=7)
        # [x_grid, y_grid] = np.meshgrid(-np.linspace(0.0, 1.8, 7), -np.linspace(0.0, 1.8, 7))
        # MOVE_X = x_grid.ravel()
        # MOVE_Y = y_grid.ravel()
        # print "The x_grid is ", MOVE_X
        # print "The y_grid is ", MOVE_Y
        # Change the move_x and move_y here for a line
        # P_line_v_tform, P_Stereo_grid_tform = self.MapMtiStereo()
        # MOVE_X = P_Stereo_grid_tform[:,0]
        # MOVE_Y = P_Stereo_grid_tform[:,1]

    def Scan5d(self):

        # Read the template image
        img_template = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png')
        [h, w, _] = img_template.shape
        print "The height of the image is ", h
        print "The width of the image is ", w

        # Define the ROI
        # r = cv2.selectROI(img_template)
        r = (292.0, 173.0, 144.0, 142.0)
        # imCrop = img_template[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # cv2.imshow("Image", imCrop)
        # cv2.waitKey(0)
        # print "The region is ", r
        # print "The r shape is ", r.shape
        # apply the mask to the image

        mask = np.zeros(img_template.shape[:2], np.uint8)
        mask[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = 255
        img_masked = cv2.bitwise_and(img_template, img_template, mask = mask)
        cv2.imshow("Maksed Image", img_masked)
        cv2.waitKey(0)

        # Apply the Harris Detector to this region
        img_mask_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        img_mask_gray = np.float32(img_mask_gray)
        dst = cv2.cornerHarris(img_mask_gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.001 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        img_masked[dst > 0.001 * dst.max()] = [0, 0, 255]

        cv2.imshow('dst', img_masked)
        cv2.waitKey(0)

        corners_org_y, corners_org_x = np.where(dst > 0.001 * dst.max())
        print "The corners org are -- x ", corners_org_x.shape
        print "The corners org are -- y ", corners_org_y.shape

        corners_org = np.zeros([len(corners_org_x), 2])
        corners_org[:, 0] = corners_org_x
        corners_org[:, 1] = corners_org_y

        # corners_org = np.hstack([corners_org_x.transpose(), corners_org_y.transpose()])
        print "The corners org is ", corners_org

        connectivity = 8
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity, cv2.CV_32S)
        # print "The centroids are ", centroids
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(img_mask_gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # use the original points or the detectors
        corners = corners_org

        # Threshold the data
        idx_corners_use = np.where((corners[:, 0] > (r[0] + 5)) & (corners[:, 0] < (r[0] + r[2] - 5)))
        print "The index of the new data is ", idx_corners_use
        corners = corners[idx_corners_use]
        # centroids = centroids[idx_corners_use]
        # remove the first value -- the noisy data -- it depends on the image
        corners = corners[1:]
        print "The new corners is ", corners

        # res = np.hstack((centroids, corners))
        # res = centroids
        res = np.int0(corners)
        res = np.int0(res)
        img_masked[res[:, 1], res[:, 0]] = [255, 0, 0]
        print "The res is ", res

        # Test the new image
        img_new = np.zeros((h, w, 3), np.uint8)
        img_new[res[:, 1], res[:, 0]] = [255, 0, 0]
        cv2.imshow('image template', img_new)
        cv2.waitKey(0)

        # img_masked[res[:, 3], res[:, 2]] = [0, 255, 0]
        # print "The points of the corner coordinates are ", centroids.shape
        # print "The points of the corners are ", corners

        flag_savenpy = raw_input("Do you want to save the point data ?")
        if flag_savenpy == 'yes':
            np.save('P_STEREO_2d.npy', corners)

    def MapMtiStereo(self):

        # Map the stereo vision as well as the region

        # Read MTI 3D
        P_MTI_3d = loadmat('C:\Users\gm143\TumorCNC_brainlab\BTL\P_MTI_3d.mat')
        P_MTI_3d = P_MTI_3d['P_MTI_3d']
        print "The MTI 3D data is ", P_MTI_3d

        # Read the Stereo 3D
        P_Stereo_3d = loadmat('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_3d.mat')
        P_Stereo_3d = P_Stereo_3d['P_STEREO_3d']
        print "The Stereo 3D data is ", P_Stereo_3d

        # Read the 3D coordinates of the shown data (The data that requires to be shown on the plane
        P_Stereo_target = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_target.npy')
        P_Stereo_target = np.multiply(P_Stereo_target, 100)
        P_Stereo_target = P_Stereo_target[np.logical_and(P_Stereo_target[:,0] <> 0, P_Stereo_target[:,1] <> 0)]
        print "The Stereo grid is ", P_Stereo_target

        # Apply the transformation
        R_final = np.asarray([[0.99924, -0.0060221, -0.03842],
                               [0.0072509, -0.94176, 0.3362],
                              [-0.038208, -0.33622, -0.94101]])
        t_final = np.asarray([3.6437, 6.2995, 47.132])

        fixed = P_MTI_3d
        moving = P_Stereo_3d
        moving_tform = np.matmul(moving, R_final) + np.tile(t_final, (len(moving), 1))
        P_Stereo_target_tform = np.matmul(P_Stereo_target, R_final) + np.tile(t_final, (len(P_Stereo_target), 1))

        # Viz the result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P_Stereo_3d[:, 0], P_Stereo_3d[:, 1], P_Stereo_3d[:, 2], c='b', marker='^')
        ax.scatter(P_MTI_3d[:, 0], P_MTI_3d[:, 1], P_MTI_3d[:, 2], c='r', marker='o')
        ax.scatter(moving_tform[:, 0], moving_tform[:, 1], moving_tform[:, 2], c='g', marker='o')
        ax.scatter(P_Stereo_target_tform[:, 0], P_Stereo_target_tform[:, 1], P_Stereo_target_tform[:, 2], c='b', marker='^')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

        P_Stereo_grid_tform = np.multiply(P_Stereo_target_tform, 0.3937)
        print("The stereo grid transformation is ", P_Stereo_target_tform)
        P_Stereo_target_tform[:,0] = -P_Stereo_target_tform[:,0]
        print("The stereo grid transformation is ", P_Stereo_target_tform)

        return P_Stereo_target_tform

        # Test a line
        # Stereo to MTI
        # p_h_1 = np.asarray([-1.1915, -0.23311, 25.063])
        # p_h_2 = np.asarray([1.1925, -0.23389, 25.026])
        # p_v_1 = np.asarray([0.003205, -1.374, 24.672])
        # p_v_2 = np.asarray([0.005735, 0.8826, 25.371])
        #
        # p_x_v = np.linspace(p_v_1[0], p_v_2[0], num=20)
        # p_y_v = np.linspace(p_v_1[1], p_v_2[1], num=20)
        # p_z_v = np.linspace(p_v_1[2], p_v_2[2], num=20)
        #
        # P_line_v = np.zeros((len(p_x_v), 3))
        # P_line_v[:,0] = p_x_v
        # P_line_v[:,1] = p_y_v
        # P_line_v[:,2] = p_z_v
        #
        # P_line_v_tform = np.matmul(P_line_v, R_final) + np.tile(t_final, (len(P_line_v), 1))

    def StereoROITarget(self):

        # Capture a realtime image
        color_image = self.CaptureMaskImg()

        # Get the 2D coordinates for the target points
        # P_target_2D = self.MakeRandomLine()
        P_target_2D = PixelToRegion()
        # P_target_2D = self.GUIClickPoint()
        # print "The shape of P_target_2D is", P_target_2D.shape

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe_profile = pipeline.start(config)
        curr_frame = 0

        # Generate the 3D point cloud and save to the np format
        color_idx = []
        curr_frame = 0
        try:
            while (curr_frame < 20):
                frames = pipeline.wait_for_frames()
                align_to = rs.stream.color
                align = rs.align(align_to)
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
                curr_frame += 1
        finally:
            pipeline.stop()

        img_template = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png')
        r = (292.0, 173.0, 144.0, 142.0)

        # Determine the region of the target region
        delta_region = 100
        x1_cols = np.int(r[0]) - delta_region
        x2_cols = np.int(r[0]) + np.int(r[2]) + delta_region
        y1_rows = np.int(r[1]) - delta_region
        y2_rows = np.int(r[1]) + np.int(r[3]) + delta_region
        Range = (x1_cols, x2_cols, y1_rows, y2_rows)
        print "The range of the target region is ", Range

        # init the cols and rows
        cols = list(range(x1_cols, x2_cols))
        rows = list(range(y1_rows, y2_rows))

        h, w, d = img_template.shape
        img_use = np.zeros((h, w, d), np.uint8)
        # print "The STEREO points are ", P_STEREO_2d
        for i in range(len(P_target_2D)):
            img_use[P_target_2D[i, 1], P_target_2D[i, 0], :] = [255, 0, 0]
            print "Finish the %d point " % i
        print"The P_line is ", P_target_2D

        count = 0
        count_grid = 0
        count_plane = 0
        Pc = np.zeros((len(rows) * len(cols), 3))
        Color_Vec = np.zeros((len(rows) * len(cols), 3))
        Pc_target = np.zeros((len(P_target_2D), 3))
        Color_Vec_target = np.zeros((len(P_target_2D), 3))

        for i in rows:
            for j in cols:

                # Get the distance of the specific pixel coordinate
                depth = depth_frame.get_distance(j, i)

                # Get the point coordinate of the specific region
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [j, i], depth)
                color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
                pixel_point = rs.rs2_project_point_to_pixel(color_intrin, color_point)

                # Save the data point -- save the point to the region
                j_idx = pixel_point[0]
                i_idx = pixel_point[1]

                if (np.isnan(i_idx) or np.isnan(j_idx) or j_idx > 640 or i_idx > 480):
                    i_idx = 0
                    j_idx = 0

                # Get the corresponding color vectors
                colorvec_target = img_use[np.int(i_idx), np.int(j_idx), :]

                # if the point is a red dot
                if (colorvec_target[0] == 255 and colorvec_target[1] == 0 and colorvec_target[2] == 0):
                    # color_idx.append(count_grid)
                    print(count_grid)
                    Pc_target[count_grid, :] = depth_point
                    Color_Vec_target[count_grid, :] = colorvec_target
                    count_grid += 1

                colorvec_plane = img_template[np.int(i_idx), np.int(j_idx), :]

                # if the light is not in 0,0,0
                if (colorvec_plane[0] != 0 and colorvec_plane[1] != 0 and colorvec_plane[2] != 0):
                    # color_idx.append(count_plane)
                    # print(count_plane)
                    Pc[count_plane, :] = depth_point
                    Color_Vec[count_plane, :] = colorvec_plane
                    count_plane += 1
                count += 1

        print "The points of the target points are", Pc_target
        print "The shape of the target points is ", Pc_target.shape

        flag_save = raw_input("Do you want to save the data (ROI)?")
        if flag_save == 'yes':
            # np.save('P_STEREO_3d.npy', Pc_grid)
            np.save('P_STEREO_ROI.npy', Pc)
            np.save('P_STEREO_ROI_Color.npy', Color_Vec)
        flag_save = raw_input("Do you want to save the target 3D data?")
        if flag_save == 'yes':
            np.save('P_STEREO_target.npy', Pc_target)
            np.save('P_STEREO_target_Color.npy', Color_Vec_target)

        # Viz the colorized point cloud
        Actor_grid = BTL_VIZ.ActorNpyColor(Pc_target, Color_Vec_target)
        # Actor_plane = BTL_VIZ.ActorNpyColor(Pc, Color_Vec)
        VizActor([Actor_grid])
        # vtk_Pc = npy2vtk(Pc_grid)
        # BTL_VIZ.VizVtk([vtk_Pc])

    def MakeRandomLine(self):

        color_image = self.CaptureMaskImg()

        # Make a random line
        # pick up two points
        p_1 = np.asarray([345, 260])
        p_2 = np.asarray([384, 224])
        p_line_x = np.linspace(p_1[0], p_2[0], 100)
        p_line_y = np.linspace(p_1[1], p_2[1], 100)
        P_line = np.zeros((len(p_line_x), 2))
        P_line[:, 0] = p_line_x
        P_line[:, 1] = p_line_y
        P_line = np.int_(P_line)

        # Generate the line -- show on the image
        cv2.line(color_image, (p_1[0], p_1[1]), (p_2[0], p_2[1]), (0, 255, 0), thickness=3)
        for points in P_line:
            print(points)
            cv2.circle(color_image, tuple(points), 1, (0, 0, 255))
        # cv2.imshow('color image', color_image)
        # cv2.waitKey(0)

        print "The shown points are ", P_line

        return P_line

    def GUIClickPoint(self):

        # Design a userinterface to choose a point
        root = Tk()
        # setting up a tkinter canvas with scrollbars
        frame = Frame(root, bd=2, relief=SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = Scrollbar(frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=E + W)
        yscroll = Scrollbar(frame)
        yscroll.grid(row=0, column=1, sticky=N + S)
        canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky=N + S + E + W)
        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)
        frame.pack(fill=BOTH, expand=1)

        # adding the image
        # File = askopenfilename(parent=root, initialdir="C:/", title='Choose an image.')
        File = 'C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png'
        img = PIL.ImageTk.PhotoImage(PIL.Image.open(File))
        # print "The image is ", img
        canvas.create_image(0, 0, image=img, anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))

        img_show = cv2.imread(File)

        # function to be called when mouse is clicked
        P_centers = []
        def printcoords(event):
            # outputting x and y coords to console
            # move the mouse to the region
            cv2.circle(img_show, tuple([event.x, event.y]), 2, (0, 0, 255))
            cv2.imshow('img_show', img_show)
            cv2.waitKey(10)
            P_centers.append([event.x, event.y])
            print (event.x, event.y)

        # mouseclick event
        canvas.bind("<Button 1>", printcoords)
        root.mainloop()

        # Print the collected points
        print(P_centers)

        # Show the image and a set of points
        for points in P_centers:
            print(points)
            cv2.circle(img_show, tuple(points), 2, (0, 0, 255))

        P_target_2D = np.zeros((len(P_centers), 2))
        for index, elem in enumerate(P_centers):
            P_target_2D[index,:] = elem
        P_target_2D = np.int_(P_target_2D)
        print(P_target_2D)

        cv2.imshow('img_show', img_show)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        return P_target_2D

    # def pnp(self):
        #
        #     # Configure the realsense device
        #     pipeline = rs.pipeline()
        #     config = rs.config()
        #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #     pipeline.start(config)
        #
        #     # pnp operation to calculate the result
        #     # Intrinsic parameters
        #     frames = pipeline.wait_for_frames()
        #     color_frame = frames.get_color_frame()
        #     color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        #
        #     # The camera matrix
        #     # width: 640, height: 480, ppx: 312.14, ppy: 238.018, fx: 619.846, fy: 619.846, model: None
        #     camera_matrix = np.array(
        #         [[619.846, 0, 312.14],
        #          [0, 619.846, 238.018],
        #          [0, 0, 1]], dtype="double"
        #     )
        #     print(camera_matrix)
        #
        #     # Read the 2D points
        #     P_2D = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_2D.npy')
        #     print(P_2D.shape)
        #
        #     # Read the 3D points
        #     p3D = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_3D.npy')
        #     P_3D = p3D[1:]
        #     print(P_3D)
        #
        #     # normalize the 3D points to the origin of the first point
        #     P_3D_new = P_3D - np.tile(P_3D[0,:], (270, 1))
        #     print(P_3D_new)
        #
        #     # PNP to find the relation and transformation
        #     # Assume no distortion
        #     dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        #     (success, rotation_vector, translation_vector) = cv2.solvePnP(P_3D_new, P_2D, camera_matrix,
        #                                                                   dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #     rmat = cv2.Rodrigues(rotation_vector)[0]
        #     print "Rotation Vector:\n {0}".format(rotation_vector)
        #     print "Translation Vector:\n {0}".format(translation_vector)
        #     print "The rotation matrix is ", rmat
        #
        #     # Test the 3D point
        #     test_3D = P_3D_new  # after normalization
        #     test_2D = P_2D
        #
        #     print "The test_3D is ", test_3D
        #     print "The test 2D is ", test_2D
        #
        #     (test_point2D, jacobian) = cv2.projectPoints(np.array([test_3D]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        #
        #     # Reshape the test_point2D (L, L, L) to (L, L)
        #     test_point2D = np.reshape(test_point2D, (len(test_point2D), 2))
        #
        #     print("The test 2D point is ", test_point2D)
        #     print("The shape of test 2D point is ", test_point2D.shape)
        #
        #     # Calculate the average of the error (labelled as pixel values)
        #     diff = test_2D - test_point2D
        #     mTRE = (np.sum(np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1])))) / len(test_point2D)
        #     print("The diff of 2D image points are", diff.shape)
        #     # mean target registration error
        #     print("The mTRE (2D image) is ", mTRE)
        #
        #     return rmat, translation_vector, mTRE
        #
        # def VizPNP(self):
        #
        #     # Viz the two coordinate system
        #     renderer = vtk.vtkRenderer()
        #     renderWindow = vtk.vtkRenderWindow()
        #     renderWindow.AddRenderer(renderer)
        #     renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        #     renderWindowInteractor.SetRenderWindow(renderWindow)
        #     renderer.SetBackground(.1, .2, .3)  # Background dark blue
        #
        #     transform_1 = vtk.vtkTransform()
        #     transform_1.Translate(0.0, 0.0, 0.0)
        #     transform_1.RotateX(45)
        #
        #     transform_2 = vtk.vtkTransform()
        #     transform_2.Translate(1.0, 1.0, 1.0)
        #
        #     # Design the first axes
        #     axes_1 = vtk.vtkAxesActor()
        #     axes_1.SetUserTransform(transform_1)
        #     renderer.AddActor(axes_1)
        #
        #     axes_2 = vtk.vtkAxesActor()
        #     axes_2.SetUserTransform(transform_2)
        #     renderer.AddActor(axes_2)
        #
        #     renderer.ResetCamera()
        #     renderWindow.Render()
        #     # begin mouse interaction
        #     renderWindowInteractor.Start()

    # def Test_1(self):
    #     # Apply the transformation
    #     R_final = np.asarray([[-0.99999, 0.0053327, 0.00040876],
    #                           [0.0048848, 0.94176, - 0.33624],
    #                           [0.002178, 0.33623, 0.94178]])
    #     t_final = np.asarray([-2.7387, -10.872, -0.15546])
    #     # Test the real data
    #     P_STEREO_ROI = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_ROI.npy')
    #     P_STEREO_ROI_Color = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_ROI_Color.npy')
    #
    #     P_MTI_3d = loadmat('C:\Users\gm143\TumorCNC_brainlab\BTL\P_MTI_3d.mat')
    #     P_MTI_3d = P_MTI_3d['P_MTI_3d']
    #     print "The MTI 3D data is ", P_MTI_3d
    #
    #     # remove the zero values
    #     idx_remove = np.int_(np.where((P_STEREO_ROI[:,0] == 0) & (P_STEREO_ROI[:,1] == 0) & (P_STEREO_ROI[:,2] == 0)))
    #     # P_STEREO_ROI = P_STEREO_ROI[~idx_remove[0], :]
    #     # P_STEREO_ROI_Color = P_STEREO_ROI_Color[~idx_remove[0], :]
    #     P_STEREO_ROI = np.multiply(P_STEREO_ROI, 100)
    #     print "The index of the removal values are ", idx_remove
    #     print "The ROI is ", np.max(P_STEREO_ROI)
    #     # Actor_grid = BTL_VIZ.ActorNpyColor(P_STEREO_ROI, P_STEREO_ROI_Color)
    #     vtk_Pc = npy2vtk(P_STEREO_ROI)
    #     BTL_VIZ.VizVtk([vtk_Pc])
    #     Actor_plane = BTL_VIZ.ActorNpyColor(P_STEREO_ROI, P_STEREO_ROI_Color)
    #     VizActor([Actor_plane])
    #
    #     # Transform the test data
    #     P_STEREO_ROI_tform = np.matmul(P_STEREO_ROI, R_final) + np.tile(t_final, (len(P_STEREO_ROI), 1))
    #     vtk_Pc_tform = npy2vtk(P_STEREO_ROI_tform)
    #     vtk_MTI_3d = npy2vtk(P_MTI_3d)
    #     BTL_VIZ.VizVtk([vtk_Pc_tform])
    #     Actor_MTI = BTL_VIZ.ActorNpyColor(P_MTI_3d, P_STEREO_ROI_Color[0:len(P_MTI_3d), :])
    #     Actor_plane_tform = BTL_VIZ.ActorNpyColor(P_STEREO_ROI_tform, P_STEREO_ROI_Color)
    #     VizActor([Actor_plane, Actor_plane_tform, Actor_MTI])

def Inch2Direction(unit_move, move_x, move_y):

    # move_x: inches on x axis
    # mvoe_y: inches on y axis
    # UserIn: inches on z axis

    # unit_move = 0.05    # how many steps to be moved
    steps_x = move_x / unit_move
    steps_y = move_y / unit_move
    sign_x = np.int(np.sign([steps_x]))
    sign_y = np.int(np.sign([steps_y]))
    UserIn = []

    if sign_x == -1:  # negative
        print "test"
        for i in range(np.int(np.absolute(steps_x))):
            UserIn.append('a')
    if sign_x == 1:   # positive
        for i in range(np.int(np.absolute(steps_x))):
            UserIn.append('d')
    if sign_y == -1:
        for i in range(np.int(np.absolute(steps_y))):
            UserIn.append('s')
    if sign_y == 1:
        for i in range(np.int(np.absolute(steps_y))):
            UserIn.append('w')

    return UserIn

def timer(name, delay, repeat):
    print "Timer: " + name + " Started"
    while repeat > 0:
        time.sleep(delay)
        print name + ": " + str(time.ctime(time.time()))
        repeat -= 1
    print "Timer: " + name + " Completed"

def PixelToRegion():
    # input the pixel and return the region within the pixel centers
    # input: a set of pixel coordinates
    # output: a set of points related to the region -- list and numpy inside
    p_center = np.asarray([325, 206])
    radius = 10
    img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png')
    mask = np.zeros(img.shape, np.uint8)
    print "image shape", img.shape
    cv2.circle(mask, tuple([p_center[0], p_center[1]]), radius, 255, -1)
    cv2.circle(img, tuple([p_center[0], p_center[1]]), radius, 255, -1)
    where = np.where(mask == 255)
    # where[0] -- y/h where[1] -- x/h
    intensity_values_from_original = img[where[1], where[0]]        # x and y order

    # print "The intensity coordinates are ", where[1]

    centers = np.zeros((len(where[0]), 2))
    centers[:, 0] = where[1]
    centers[:, 1] = where[0]
    centers = np.int_(centers)
    print "The centers are ", centers
    print "The shape of the center region is ", centers.shape

    # np.save('P_STEREO_2d.npy', centers)
    # cv2.imshow('img_circle', img)
    # cv2.waitKey(0)

    return centers

if __name__ == "__main__":

    test = PNP()

    # Test LaserDetect function -- finished
    # test.LaserSpotDetect()

    # Test the cameraintrinsic -- finished
    # test.CamIntrinsic()

    # Test the RsCapture -- finished
    # test.RsRealtime()

    # Test the CapRealtime -- finished
    # test.CapRealtime()

    # Test MoveOrigin -- finished
    # test.MoveOrigin()

    # Test MovePoint -- finished
    # move_x = 0.5 # inches
    # move_y = -0.5 # inches
    # test.MovePoint(move_x, move_y)

    # Test ScanPoints -- finished
    # test.ScanPoints()

    # Test Extract2DPoints -- finished
    # test.LaserSpotCalibration2D()

    # Test CaptureMaskImg -- finished
    # test.CaptureMaskImg()

    # Test MapMtiStereo
    # test.MapMtiStereo()

    # Test Scan3d -- finished
    # P_Stereo_target_tform = test.MapMtiStereo()
    # MOVE_X = P_Stereo_target_tform[:, 0]
    # MOVE_Y = P_Stereo_target_tform[:, 1]
    # test.Scan3d(MOVE_X, MOVE_Y)

    # MakeRandomLine
    # test.MakeRandomLine()

    # Test StereoROITarget -- finished
    # test.StereoROITarget()
