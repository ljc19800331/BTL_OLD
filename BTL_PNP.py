# Calibration PNP
# ref: # https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
# https://github.com/IntelRealSense/librealsense/issues/869
# https://www.youtube.com/watch?v=NtJXi7u_fJo
# ROI design for the figure

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
from controlLaser import ControlLaser
import BTL_FastPc
import matlab.engine

class PNP:

    def __init__(self):

        self.AC = AngleConvert()
        self.CM = ControlMovement()
        self.ST = ScanTypes()
        self.CS = ConvertSteering()
        self.CD = ControlDAQ()
        self.GSA = GenerateScanAngles()
        # self.CL = ControlLaser(port="COM8")
        # self.CL = ControlLaser()
        # self.CLco2 = ControlLaser(port="COM6")

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

    def ShowOneImg(self, path):
        img = cv2.imread(path)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    def LaserSpotDetect(self):

        # Define the ROI manually
        img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
        [height, width, dim] = img.shape
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
                    cv2.circle(color_image, Spot_center, 30, (0, 0, 0), 2)
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Color image', color_image)
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

        unit_move = 0.01
        UserIn = Inch2Direction(unit_move, move_x, move_y)

        flag_hene = raw_input("Do you want to include hene? (yes/no)")
        if flag_hene == 'yes':
            UserIn_hene = Inch2Direction(0.01, -0.42, -0.11)
            UserIn = UserIn_hene + UserIn
        centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn)

    def MoveFromAngle(self, x_angle, y_angle):

        x_cent_ang = x_angle  # np.array( -1.0381408);
        y_cent_ang = y_angle  # np.array(-1.168893);
        self.CD.point_move(x_cent_ang, y_cent_ang, .005)

    def MoveFromPosition(self, x_position, y_position, z_position):

        [x_ang, y_ang] = self.ST.CS.angular_position(x_position, y_position, z_position)
        self.CD.point_move(x_ang, y_ang, 0.005)

    def MoveTest(self):

        # Move to the origin to restart
        self.MoveOrigin()

        # calculate the angles
        # x_ang = [0]
        # y_ang = [0]

        # x = np.array([-0.03505])
        # y = np.array([0.06479])
        # z = np.array([9.271])

        x = np.array([-0.043356])
        y = np.array([0.055868])
        z = np.array([9.2782])

        [x_ang_mti, y_ang_mti] = self.CS.angular_position(x, y, z)
        print "The angle of mtiis ", x_ang_mti, y_ang_mti

        # x_ang_mti = - 0.82
        # y_ang_mti = - 0.97

        # Why we need to input the first z distance?
        self.CD.point_move(x_ang_mti, y_ang_mti, .005)
        z_dist_vol = float(self.CD.point_measure(1))                # Measure the plane -- voltage
        z_dist = float((z_dist_vol / 0.04 + 175) * 0.03937007874)   # convert voltage to the distance
        print "The distance after point move is ", z_dist

        # z_dist = 9.25417358348                                    # This distance is really important
        # targetPoints = self.CS.xy_position(x_ang_mti, y_ang_mti, z_dist, 0)
        targetPoints = self.ST.measure_point_wes(x_ang_mti, y_ang_mti)
        print "The target point of mti (measure point wes) is ", targetPoints

        x_co2 = np.array([targetPoints[0, 0]])
        y_co2 = np.array([targetPoints[0, 1]])
        z_co2 = np.array([targetPoints[0, 2]])

        print(x_co2, y_co2, z_co2)

        [x_ang_co2, y_ang_co2] = self.CS.angular_position_co2(x_co2, y_co2, z_co2)
        print "The angle of co2 is ", x_ang_co2, y_ang_co2
        self.ST.CD.point_move(x_ang_co2[0], y_ang_co2[0], 1.)

        # input the x,y,z coordinates
        # print('here')
        # x = np.array([0.29850429])
        # y = np.array([0.38040736])
        # z = np.array([9.27711944])

        # [[0.29850429  0.38040736  9.27711944]]
        # [[ 0.22643021  0.00724624  2.53248866]]
        # [[0.23437492  0.01425146  2.54887697]]

        # print('here 2')
        # # [x_ang, y_ang] = self.CS.angular_position(x, y, z)
        # [x_ang, y_ang] = self.CS.angular_position(x, y, z)
        # print(x_ang, y_ang)
        #
        # x_ang[0] = 0
        # y_ang[0] = 0
        # zdist = 16.0095854846
        #
        # # Measure the distances in this point
        # targetPoints = self.CS.xy_position(x_ang, y_ang, zdist, 0)
        # print "The target points are ", targetPoints
        #
        # # mvoe the target to this point
        # self.ST.CD.point_move(x_ang[0], y_ang[0], 1.)
        # print('here 3')

        # print("Tickling laser...")
        # ST.CL.write_message("t")
        # print "Cutting!"
        # ST.CD.point_move(x_ang[0], y_ang[0], 1.)  # move the mirrors to the start of the scan
        # msg = "p,100,100"
        # ST.CL.write_message(msg)
        # ST.CL.write_message("off")
        # exit(1)

    def MoveRegion(self):
        # input a set of 3D point region
        # move the Hene to the target points
        a = 1

    def ScanPoints(self):

        self.ST.CL.write_message("mti on")
        # self.ST.CLRedLasers.write_message("hene off")

        filename = 'ScanAngles.npy'
        if os.path.exists(filename):
            os.remove(filename)

        mtiScanSpan = [1.0, 1.0]        # The scanning region of the MTI
        pointDistance = 0.05             # inches between each point on the scan

        # Move the center with x_angle == 0 and y_angle == 0
        print 'Set center of scan.'
        x_c = 0
        y_c = 0                             # The starting angles value -- origin

        # Define the Hene coordinates
        hene_UserIn = Inch2Direction(0.05, -0.45, -0.1)
        print "The hene usein is ", hene_UserIn

        # Define the initial position from the origin -- different from (0,0)
        flag_hene = raw_input("Is the hene showing? (yes/no)")
        if flag_hene == 'yes':
            UserIn = Inch2Direction(0.05, 1.00, -1.00)
            UserIn_all = hene_UserIn + UserIn
        elif flag_hene == 'no':
            UserIn = Inch2Direction(0.05, 0.98, -0.98)
            UserIn_all = UserIn

        flag = raw_input("Do you want to move to the predefined center (yes/no)?")

        if flag == 'yes':
            centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn_all)
        if flag == 'no':
            centerScan, z_dist, center_angles = self.CM.controlPoint([x_c], [y_c])

        print "The z_dist after control point is ", z_dist
        print "The center angles are", center_angles
        print "The centerScan from ControlPoint is ", centerScan
        print 'Set scan box location and size.'

        centerScan, scanSpread, z_dist = self.CM.controlScanBox(centerScan, z_dist, mtiScanSpan, 'box', 'vertical', '2')
        scanCenter = centerScan
        print "The centerScan from Scanbox is", centerScan
        print "The scanSpread is", scanSpread

        # z_dist2 = z_dist
        pointsPerX = scanSpread[0] / pointDistance
        pointsPerY = scanSpread[1] / pointDistance
        print "The pointsPerX is ", pointsPerX
        print "The pointsPerY is ", pointsPerY

        # print 'Performing initial raster scan.'
        print "The z_dist before the input for the raster_scan is  ", z_dist
        target_points, filtered_points, x_angles, y_angles = self.ST.raster_scan_v1(centerScan, scanSpread, z_dist, pointsPerX, pointsPerY, 'Scan', 'ScanAngles')
        self.ST.plot_scan(target_points, 1)
        self.PointSTA(target_points)

        flag_save = raw_input("Do you want to save the scanning data (yes/no)?")
        if flag_save == 'yes':
            np.save('C:/Users/gm143/TumorCNC_brainlab/BTL/Pc_Phantom.npy', target_points)

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

    def LaserSpotCapture2D(self):

        # Extract the 2D coordinates from the images
        path, dirs, files = next(os.walk("C:/Users/gm143/TumorCNC_brainlab/BTL/BTL_data/ImgLaser/"))
        img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser/P_1.png')
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The region is ", r

        P_LaserCalibrate_2D = np.zeros([  len(range(1,len(files)+1)), 2 ])
        idx = 0

        for i in range(1, len(files) + 1):
            img_name = 'C:/Users/gm143/TumorCNC_brainlab/BTL/BTL_data/ImgLaser/' + 'P_' + str(i) + '.png'
            color_img = cv2.imread(img_name)
            ROI_image = color_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            idx_center_region = np.int_(np.where(grey_image > 240))
            target_val = np.int_([np.mean(idx_center_region[1]), np.mean(idx_center_region[0])])
            Spot_center = (target_val[0] + np.int(r[0]), target_val[1] + np.int(r[1]))
            # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
            # Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
            P_LaserCalibrate_2D[idx, :] = Spot_center
            # cv2.circle(color_img, tuple([Spot_center[0], Spot_center[1]]), 1, (255, 0, 0))
            # cv2.imshow('image template', color_img)
            # cv2.waitKey(500)
            print "Finish the %s image " % i
            print "The spotcenter is ", Spot_center
            idx += 1

        print "The 2D coordinates are ", P_LaserCalibrate_2D
        print "The shape of the 2D coordinates are ", P_LaserCalibrate_2D.shape

        flag_save = raw_input("Save the 2D coordinates for all the points (yes/no)?")
        if flag_save == 'yes':
            np.save('C:\Users\gm143\TumorCNC_brainlab\BTL\P_LaserCalibrate_2D.npy', P_LaserCalibrate_2D)

    def CaptureTemplateImg(self, L, W):

        self.ST.CL.write_message("mti off")

        # Configure depth and color streams
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


        flag_save = raw_input("Do you want to save the template image?")

        cap = cv2.VideoCapture(1)
        cap.set(3, L)
        cap.set(4, W)

        # Start streaming
        # pipe_profile = pipeline.start(config)
        curr_frame = 0

        try:
            while (curr_frame < 20):

                # Get the color frame
                # frames = pipeline.wait_for_frames()
                ret, frames = cap.read()

                # Align_depth_frame is a 640x480 depth image
                # align_to = rs.stream.color
                # align = rs.align(align_to)
                # aligned_frames = align.process(frames)
                # aligned_depth_frame = aligned_frames.get_depth_frame()
                # color_frame = aligned_frames.get_color_frame()
                # color_image = np.asanyarray(color_frame.get_data())

                color_image = frames

                # This is important since the first frame of the color image is dark while later it is better
                curr_frame += 1
                if (curr_frame == 10) and flag_save == 'yes' and L == 640:
                    cv2.imwrite('C:/Users/gm143/Documents/MATLAB/BTL/Data/exp_1/pos_12/before/target.png', color_image)
                if (curr_frame == 10) and flag_save == 'yes' and L == 1280:
                    cv2.imwrite('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/template_1280.png', color_image)
        finally:
            cap.release()
            # pipeline.stop()


        return color_image

    def GetCentroidFromImg(self):

        # Get the centroids of the points from the images
        # Define the ROI manually
        img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\Calibration_2D\P_1.png')
        path, dirs, files = next(os.walk("C:\Users\gm143\TumorCNC_brainlab\BTL\Calibration_2D/"))
        [height, width, dim] = img.shape
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        print "Please choose the ROI"
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The rectangular region is ", r

        # Get the centroids of the images
        Centroids = []
        foldername = 'C:\Users\gm143\TumorCNC_brainlab\BTL\Calibration_2D/'

        data_miss = []  # The missing idx case

        for i in range(0, len(files)):

            # i = 2892

            print(i)
            # if(i==1 or i == 2 or i == 3):
            #     continue

            img_name = foldername + 'P_' + str(i) + '.png'
            color_image = cv2.imread(img_name)
            # print(len(color_image))
            ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            idx = np.int_(np.where(grey_image > 250))

            # idx = np.int_(np.where(grey_image == np.max(grey_image)))
            # print(idx_grey_max)

            # u_x, v_x = np.max(grey_image)
            # print(idx)
            # print(len(idx[0]))

            if(len(idx[0]) == 0):
                # data_miss.append(i)
                idx = np.int_(np.where(grey_image > 200))
                # print "Here"
                # print(idx)

            if (len(idx[0]) == 0):
                data_miss.append(i)
                continue

            target_val = np.int_([np.mean(idx[1]), np.mean(idx[0])])
            Spot_center = (target_val[0] + np.int(r[0]), target_val[1] + np.int(r[1]))

            # print "The new spot center is ", Spot_center
            # cv2.circle(color_image, tuple([Spot_center[0], Spot_center[1]]), 5, (255, 0, 0))
            # cv2.circle(grey_image, tuple([Spot_center[0], Spot_center[1]]), 5, (0, 0, 255))
            # cv2.imshow('image template', color_image)
            # cv2.waitKey(500)
            # cv2.imwrite( 'C:/Users/gm143/TumorCNC_brainlab/BTL/BTL_data/ImgLaserSpot/' + str(i) + '.png', color_image)
            Centroids.append(Spot_center)

        # list to numpy
        SpotCenters = np.zeros((len(Centroids), 2))
        print "The Centroids is ", Centroids
        for i in range(len(Centroids)):
            SpotCenters[i, :] = Centroids[i]
        SpotCenters = np.int_(SpotCenters)
        print "The spot centers are", SpotCenters
        flag_save = raw_input("Do you want to save the MTI 2d scanning data (capture from stereo camera)?")
        if flag_save == 'yes':
            np.save('P_MTI_Calibration_interpolation_2D.npy', SpotCenters)

        # Show in the image
        for points in SpotCenters:
            print(points)
            cv2.circle(img, tuple(points), 1, (0, 0, 255))

        print "The data missing is", data_miss

        # cv2.imshow('image with spots', img)
        # cv2.waitKey(0)

    def Scan3d(self, MOVE_X, MOVE_Y):

        # Function 1: Calibration
        # Function 2: Scanning
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

        # Define the Hene coordinates
        x_c = 0
        y_c = 0
        hene_UserIn = Inch2Direction(0.01, -0.42, -0.1)
        print "The hene usein is ", hene_UserIn

        # Define the predefine mti coordinates
        mti_UserIn = Inch2Direction(0.01, 1.00, -1.00)
        print "The mti userin is ", mti_UserIn

        # Define the initial position from the origin -- different from (0,0)
        # flag_henemti = raw_input("If this is hene?")
        flag_henemti = 'yes'
        if flag_henemti == 'yes':
            UserIn_all = hene_UserIn
        else:
            UserIn_all = mti_UserIn

        # flag_predefine = raw_input("Do you want to move to the predefined center (yes/no/other)?")
        flag_predefine = 'no'
        if flag_predefine == 'yes':
            centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn_all)
        if flag_predefine == 'no':
            centerScan, z_dist, center_angles = self.CM.controlPoint([x_c], [y_c])

        # Set up the initial condition
        org_targetPoints = np.asarray([centerScan[0], centerScan[1], z_dist])
        arrowMove = 0.05                                # This is self-define
        P_MTI = np.zeros([len(MOVE_X), 3])
        print "The org_centerScan is ", org_targetPoints
        print "The current MOVE_X is ", MOVE_X
        print "The current MOVE_Y is ", MOVE_Y

        # check if we want to capture the 2D images at the same time
        # flag_2d = 'yes'
        flag_2d = raw_input("Do you want to capture the 2D calibration images at the same time?")

        # Begin the Loop to move to each input coordinate
        try:
            if flag_2d == 'yes':
                cap = cv2.VideoCapture(1)

            for i in range(len(MOVE_X)):

                move_x = MOVE_X[i]
                move_y = MOVE_Y[i]

                centerScan[0] = org_targetPoints[0] + move_x
                centerScan[1] = org_targetPoints[1] + move_y

                print "The move_x is ", move_x
                print "The move_y is ", move_y
                print "The org_centerScan is ", org_targetPoints
                print "The centerScan is ", centerScan
                print "The z dist is ", z_dist

                [x_cent_ang, y_cent_ang] = self.CS.angular_position(np.array([centerScan[0]]),
                                                                    np.array([centerScan[1]]),
                                                                    np.array([z_dist]))             # recalculate the center
                self.CD.point_move(x_cent_ang, y_cent_ang, 0.005)                                   # move to the new point

                if flag_2d == 'yes':
                    count = 0
                    # collect the data
                    while(count < 15):
                        ret, frames = cap.read()
                        if count > 10:
                            img_name = 'C:/Users/gm143/TumorCNC_brainlab/BTL/Calibration_2D/P_' + str(i) + '.png'
                            cv2.imwrite(img_name, frames)
                            # print "Save the %d image" % i
                        count += 1

                z_dist = float(self.CD.point_measure(0.1))                                          # measure a new distance (this returns a voltage)
                z_dist2 = (z_dist / 0.04 + 175) * 0.03937007874                                     # convert the voltage to a distance
                targetPoints = self.CS.xy_position(x_cent_ang, y_cent_ang, z_dist2)                 # figure out the actual point in space we are
                z_dist = targetPoints[0, 2] # return that distance
                mti_x = targetPoints[0, 0]
                mti_y = targetPoints[0, 1]
                print "The current target points are ", z_dist
                P_MTI[i, :] = np.asarray([mti_x, mti_y, z_dist])
        finally:
            if flag_2d == 'yes':
                cap.release()

        # Save MTI calibration 3D
        print "The points of MTI are ", P_MTI
        flag_save_3d = raw_input("Do you want to save the MTI target points?")
        # flag_save_3d = 'no'
        if flag_save_3d == 'yes':
            np.save('P_MIT_target_3d.npy', P_MTI)

        # Save MTI calibration 2D
        flag_save_2d = raw_input("Do you want to save MTI scanning 2d?")
        # flag_save_2d = 'no'
        if flag_save_2d == 'yes':
            self.GetCentroidFromImg()

    def Scan5d(self):

        self.CaptureTemplateImg()

        # Read the template image
        img_template = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/template.png')
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
        cv2.imshow("Template Image", img_masked)
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
            np.save('P_STEREO_Calibration_2D.npy', res)

    def MapMtiStereo(self):

        # Map the stereo vision as well as the region
        # Read MTI 3D
        P_MTI_3d = loadmat('C:\Users\gm143\TumorCNC_brainlab\BTL\Others\P_MTI_3d.mat')
        P_MTI_3d = P_MTI_3d['P_MTI_3d']
        print "The MTI 3D data is ", P_MTI_3d

        # Read the Stereo 3D
        P_Stereo_3d = loadmat('C:\Users\gm143\TumorCNC_brainlab\BTL\Others\P_STEREO_3d.mat')
        P_Stereo_3d = P_Stereo_3d['P_STEREO_3d']
        print "The Stereo 3D data is ", P_Stereo_3d

        # Read the 3D coordinates of the shown data (The data that requires to be shown on the plane
        P_Stereo_target = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_target.npy')
        P_Stereo_target = np.multiply(P_Stereo_target, 100)
        P_Stereo_target = P_Stereo_target[np.logical_and(P_Stereo_target[:,0] <> 0, P_Stereo_target[:,1] <> 0)]
        print "The Stereo target is ", P_Stereo_target

        # Apply the transformation
        raw_input("Please update the matrix")

        # R_final = np.asarray([[0.9995, -0.0183, -0.0242],
        #                       [-0.0074, -0.9203, 0.3910],
        #                       [-0.0295, -0.3907, -0.9201]])
        # t_final = np.asarray([1.0603, 2.8743, 11.6170])


        R_final = np.asarray([[0.99897, -0.0018348, -0.045348],
                              [0.013816, -0.93947, 0.34236],
                              [-0.043231, -0.34263, -0.93848]])
        t_final = np.asarray([3.7108, 6.2761, 47.273])

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

        P_Stereo_target_tform = np.multiply(P_Stereo_target_tform, 0.3937)
        print("The stereo target transformation is ", P_Stereo_target_tform)
        P_Stereo_target_tform[:,0] = -P_Stereo_target_tform[:,0]
        print("The stereo target transformation is ", P_Stereo_target_tform)

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
        color_image = self.CaptureTemplateImg()

        # Get the 2D coordinates for the target points
        P_target_2D = self.MakeRandomLine()
        # P_target_2D = PixelToRegion(345, 206)
        # P_target_2D = self.GUIClickPoint()
        # P_target_2D = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_Calibration_2D.npy')
        # P_target_2D = np.int_(P_target_2D)
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

        img_template = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/template.png')
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

        color_image = self.CaptureTemplateImg()

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

    def ControlPanel(self):

        a = 1
        # CP = BTL_GUI.GUI_Wes(Tk())
        # msg = "tickle"
        # CP.co2Callback(msg)

        # Turn on the Hene
        # flag_hene = raw_input("Do you want to turn on the HENE?")
        # if flag_hene == "yes":
        #     self.CL.write_message("hene on")
        # else:
        #     self.CL.write_message("hene off")
        #
        # # Turn on the MTI
        # flag_mti = raw_input("Do you want to turn on the MTI?")
        # if flag_mti == 'yes':
        #     self.CL.write_message("mti on")
        # else:
        #     self.CL.write_message("mti off")
        #
        # # Turn on the laser
        # flag_co2 = raw_input("Which operation for the CO2 laser?")
        #
        # try:
        #     if flag_co2 == 't':
        #         fullMsg = 't'
        #     elif flag_co2 == 'off':
        #         fullMsg = "off"
        #     elif flag_co2 == 'shoot':
        #         print "The laser is shooting -- be careful"
        #         fullMsg = 't'
        #         self.CLco2.write_message(fullMsg)
        #         fullMsg = 'p,100,1000'
        #         self.CLco2.write_message(fullMsg)
        #     else:
        #         fullMsg = "off"
        # finally:
        #     fullMsg = "off"
        #     self.CLco2.write_message(fullMsg)
        #     print "The laser is off"

    def HarrisDetector(self):

        # Detect the regions for PNP problems
        # return the 2D coordinates for the grid points
        # Choose which 3D points?

        self.CaptureTemplateImg()

        # Read the template image
        img_template = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/template.png')
        [h, w, _] = img_template.shape
        print "The height of the image is ", h
        print "The width of the image is ", w

        # Define the ROI
        # r = cv2.selectROI(img_template)
        r = (292.0, 173.0, 144.0, 142.0)
        # imCrop = img_template[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        mask = np.zeros(img_template.shape[:2], np.uint8)
        mask[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = 255
        img_masked = cv2.bitwise_and(img_template, img_template, mask=mask)
        # cv2.imshow("Template Image", img_masked)
        # cv2.waitKey(0)

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
        flag_detector = raw_input("Do you want to use the original detectors (yes/no)?")
        if flag_detector == 'yes':
            corners = corners_org
        # corners = corners_org

        # Threshold the data
        idx_corners_use = np.where((corners[:, 0] > (r[0] + 5)) & (corners[:, 0] < (r[0] + r[2] - 5)))
        print "The index of the new data is ", idx_corners_use
        corners = corners[idx_corners_use]
        # centroids = centroids[idx_corners_use]
        # remove the first value -- the noisy data -- it depends on the image
        corners = corners[1:]
        print "The new corners is ", corners.shape

        # res = np.hstack((centroids, corners))
        # res = centroids
        res = np.int0(corners)
        res = np.int0(res)
        img_masked[res[:, 1], res[:, 0]] = [255, 0, 0]
        print "The res is ", res.shape

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
            np.save('P_STEREO_Calibration_2D_Harris.npy', res)

    def pnp(self):

        # Configure the realsense device
        pipeline = rs.pipeline()
        config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        # pnp operation to calculate the result
        # Intrinsic parameters
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        print "The current color intrinsic parameter is ", color_intrin

        # Read Image
        im = cv2.imread("C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/template.png")
        size = im.shape

        # 2D image points. If you change the image, you need to change vector
        image_points = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_STEREO_Calibration_2D_Harris.npy')
        image_points = np.array(image_points, dtype="double")

        # 3D model points.
        model_points = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_MIT_Calibration_3D_Harris.npy')
        model_points[:, 0] = -model_points[:, 0]
        model_points[:, 1] = -model_points[:, 1]

        model_points[:, 0] = model_points[:, 0] - min(model_points[:, 0])
        model_points[:, 1] = model_points[:, 1] - min(model_points[:, 1])
        model_points[:, 2] = model_points[:, 2] - min(model_points[:, 2])

        print "The model points are", model_points

        # Camera internals
        camera_matrix = np.array(
            [[619.846, 0, 312.14],
             [0, 619.846, 238.018],
             [0, 0, 1]], dtype="double")

        print "Camera Matrix :\n {0}".format(camera_matrix)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Translation Vector:\n {0}".format(translation_vector)

        # test_points = model_points

        test_points = np.array([
           (-0.15892, 0.13611, 0.04),
           (-0.49606, 0.13618, 0.04),
           (-0.79833, 0.15035, 0.04),
           (0.49976, 1.0973, 0.5),
           (0.81672, 1.095, 0.5),
           (1.1219, 1.1152, 0.5),
        ])

        (projected_image_points, jacobian) = cv2.projectPoints(test_points, rotation_vector, translation_vector,
                                                               camera_matrix, dist_coeffs)
        projected_image_points = np.reshape(projected_image_points, [len(projected_image_points), 2])

        print "The projoected points are ", projected_image_points
        print "The projoected points (shape)", projected_image_points.shape

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        for p in projected_image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        # Calculate the average of the error (labelled as pixel values)
        # diff = projected_image_points - test_points[0:1,:]
        # mTRE = (np.sum(np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1])))) / len(projected_image_points)
        # print("The diff of 2D image points are", diff.shape)
        # # mean target registration error
        # print("The mTRE (2D image) is ", mTRE)

        cv2.imshow("Output", im)
        cv2.waitKey(0)

    def RegionCut(self):

        try:

            # close the laser for safety concern
            self.ST.CL.write_message("off")

            # Load the original MTI data -- x,y,z raw values
            data = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_MTI_xy_modified.npy')
            x_points_raw = data[:, 0]
            y_points_raw = data[:, 1]
            z_points_raw = data[:, 2]

            # Cut the region based on results
            raw_input('Press enter to start cutting process.')
            print "Calculating angles..."

            # CO2 angles
            # method 1
            # x_points_raw = x_points_raw + 0.43
            # y_points_raw = y_points_raw - 0.08
            # [x_ang, y_ang] = self.ST.CS.angular_position(x_points_raw, y_points_raw, z_points_raw)

            # method 2
            [x_ang, y_ang] = self.ST.CS.angular_position_co2(x_points_raw, y_points_raw, z_points_raw)
            print "x_ang, y_ang (co2) = ", x_ang, ',', y_ang

            # MTI angles
            # [x_ang, y_ang] = self.ST.CS.angular_position(x_points_raw, y_points_raw, z_points_raw)
            # print "x_ang, y_ang (co2) = ", x_ang, ',', y_ang

            # turn the MTI off
            self.ST.CLRedLasers.write_message("hene on")
            self.ST.CLRedLasers.write_message("mti on")

            # Check the hene points before -- move the points to the hene
            for i in range(len(x_ang)):
                x_cent_ang = x_ang[i]
                y_cent_ang = y_ang[i]
                self.CD.point_move(x_cent_ang, y_cent_ang, 0.005)

            raw_input("Pay attention to wear glass")
            raw_input("close the tip on the whole device")

            self.ST.CLRedLasers.write_message("hene on")

            print "Cutting!"
            self.ST.CD.point_move(x_ang[0], y_ang[0], 1.)  # move the mirrors to the start of the scan
            self.ST.CL.write_message("t")
            sleep(1)

            msg = "continuous, " + str(50)
            myFrequency = 500  # bigger than 1000 causes an error, the mirrors jump.
            self.ST.CL.write_message(msg)
            self.ST.CD.cutting_scan_wes(x_ang, y_ang, myFrequency)
            self.ST.CL.write_message("off")

        finally:
            self.ST.CL.write_message("off")

        # self.ST.CL.write_message("off")
        # self.ST.CLRedLasers.write_message("hene off")
        # self.ST.CLRedLasers.write_message("mti on")
        # self.ST.CL.write_message("off")
        # self.ST.CL.close_serial()

    def RegionCut_ANN(self):

        filename = 'C:/Users/gm143/TumorCNC_brainlab/BTL/xyz_points.csv'

        # Cut the region based on the ANN model
        try:
            self.ST.CL.write_message("off")
            self.ST.CLRedLasers.write_message("hene on")
            # filename = 'xyz_points.csv'
            print "Loading file: " + str(filename)
            with open(filename, 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                a = list(spamreader)
                x_points_raw = np.zeros(len(a))
                y_points_raw = np.zeros(len(a))
                z_points_raw = np.zeros(len(a))
                for idx, row in enumerate(a):
                    aPoint = map(float, row)
                    x_points_raw[idx] = (aPoint[0])  # convert mm to inches
                    y_points_raw[idx] = (aPoint[1])
                    z_points_raw[idx] = (aPoint[2])
            asdf = np.array([x_points_raw, y_points_raw, z_points_raw])
            self.ST.plot_scan(asdf.transpose())

            raw_input("Tickling laser...")
            self.ST.CL.write_message("t")
            self.ST.CLRedLasers.write_message("mti off")
            self.ST.CLRedLasers.write_message("hene on")

            raw_input('Press enter to start cutting process.')
            print "Calculating angles..."
            [x_ang, y_ang] = self.CS.angular_position_co2(x_points_raw, y_points_raw, z_points_raw)
            print "x_ang, y_ang (co2) = ", x_ang, ',', y_ang

            # for i in range(len(x_ang)):
            #     x_cent_ang = x_ang[i]
            #     y_cent_ang = y_ang[i]
            #     self.CD.point_move(x_cent_ang, y_cent_ang, 0.005)

            raw_input('Press enter for shooting.')
            flag_laser = 1
            while(flag_laser < 2):
                self.ST.CLRedLasers.write_message("mti off")
                self.ST.CLRedLasers.write_message("hene on")
                print "Cutting!"
                self.ST.CD.point_move(x_ang[0], y_ang[0], 1.)       # move the mirrors to the start of the scan
                power = 55  # 80  70  55
                # power = raw_input('What is the input of the laser power?')
                msg = "continuous, " + str(power)
                myFrequency = 53  # 53 200    250                             # bigger than 1000 causes an error, the mirrors jump.
                self.ST.CL.write_message(msg)
                self.ST.CD.cutting_scan_wes(x_ang, y_ang, myFrequency)
                # self.ST.CL.write_message("off")

                # self.ST.CLRedLasers.write_message("hene off")
                # self.ST.CLRedLasers.write_message("mti on")
                # self.ST.CL.write_message("off")
                # self.ST.CL.close_serial()

                flag_laser += 1

                # flag_laser = raw_input("Do you want to continue? (1/0)")
                # if flag_laser == '0':
                #     break
                # print(flag_laser)
                # test.CaptureTemplateImg()

        finally:
            self.ST.CL.write_message("off")
            self.ST.CLRedLasers.write_message("hene off")

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

    def BTL_Draw(self):

        # Draw the region manually in the image -- with human loop
        # Draw a rectangular or circular region
        raw_input("remember to press m after starting")

        global ix, iy, drawing, mode
        drawing = False
        mode = True
        ix, iy = -1, -1
        P_ROI = []

        # Mouse callback function
        def draw_circle(event, x, y, flags, param):
            global ix, iy, drawing, mode

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    if mode == True:
                    #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
                    # else:
                        P_ROI.append([x, y])
                        # print(P_ROI)
                        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if mode == True:
                    cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        # img = np.zeros((512, 512, 3), np.uint8)
        img = cv2.imread('C:/Users/gm143/Documents/MATLAB/BTL/Data/exp_1/pos_12/before/target.png', 1)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        while (1):
            cv2.imshow('image', img)
            k = cv2.waitKey(1) & 0xFF
            # if k == ord('m'):
            #     mode = not mode
            if k == ord('q'):
                print(P_ROI)
                break
            elif k == 27:
                break

        cv2.destroyAllWindows()

        # mask (of course replace corners with yours)
        points = np.array([P_ROI], dtype = np.int32)

        print(points)
        print("The shape is ", img.shape)
        print("The shape is ", img.shape[0])

        h = img.shape[0]
        w = img.shape[1]

        print(h)
        print(w)

        mask = np.zeros(img_grey.shape, dtype = np.uint8)
        roi_corners = np.array(points, dtype = np.int32)  # pointsOf the polygon Like [[(10,10), (300,300), (10,300)]]
        white = (255, 255, 255)
        mask_use = cv2.fillPoly(mask, roi_corners, 255)

        # apply the mask
        masked_image = cv2.bitwise_and(img_grey, mask)

        # return the mask indices
        mask_idx = np.where(masked_image != 0)
        x_roi = mask_idx[1]
        y_roi = mask_idx[0]
        ROI = np.zeros([len(x_roi), 2])
        ROI[:,0] = x_roi
        ROI[:,1] = y_roi
        ROI = np.array(ROI, dtype = np.int32)
        print(ROI)

        # display your handywork
        cv2.imwrite('C:/Users/gm143/Documents/MATLAB/BTL/Data/exp_1/pos_12/before/mask.png', mask)
        cv2.imshow('masked image', masked_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return ROI

    def GetMtiTexColor(self, P_ROI):

        # Read the data
        MTI_tex_1 = np.load('C:/Users/gm143/TumorCNC_brainlab/BTL/MTI_textures.npy')
        MTI_vtx_1 = np.load('C:/Users/gm143/TumorCNC_brainlab/BTL/MTI_vertices.npy')
        img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/target.png')
        [h, w, d] = np.asarray(img.shape)

        # Remove the 0 values
        idx_remove = np.where((MTI_vtx_1[:, 0] == 0) & (MTI_vtx_1[:, 1] == 0) & (MTI_vtx_1[:, 2] == 0))
        mask = np.ones(MTI_vtx_1[:, 0].shape, dtype = bool)
        mask[idx_remove] = False
        MTI_vtx_2 = MTI_vtx_1[mask]
        MTI_tex_2 = MTI_tex_1[mask]

        # Texture to pixel
        MTI_tex_2[:, 0] = np.asarray(MTI_tex_2[:, 0] * w, dtype=np.int32)
        MTI_tex_2[:, 1] = np.asarray(MTI_tex_2[:, 1] * h, dtype=np.int32)

        # Remove the out of range index
        idx_remove = np.where(
            (MTI_tex_2[:, 0] >= w) | (MTI_tex_2[:, 0] <= 0) | (MTI_tex_2[:, 1] >= h) | (MTI_tex_2[:, 1] <= 0))
        mask = np.ones(MTI_tex_2[:, 0].shape, dtype = bool)
        mask[idx_remove] = False
        MTI_vtx_3 = MTI_vtx_2[mask]
        MTI_tex_3 = MTI_tex_2[mask]

        MTI_vtx = np.asarray(MTI_vtx_3)
        MTI_tex = np.asarray(MTI_tex_3, dtype = np.int32)

        # Color vector
        colorvec = np.zeros([len(MTI_tex[:, 0]), 3])
        for i in range(len(MTI_tex[:, 0])):
            idx_width = MTI_tex[i, 0]
            idx_height = MTI_tex[i, 1]
            rgb = img[idx_height, idx_width, :]
            colorvec[i, 0] = rgb[0]
            colorvec[i, 1] = rgb[1]
            colorvec[i, 2] = rgb[2]

        colorvec = np.asarray(colorvec, dtype = np.int32)

        # Viz the colorized point cloud results
        obj_actor = BTL_VIZ.ActorNpyColor(MTI_vtx, colorvec)
        BTL_VIZ.VizActor([obj_actor])

        # Colorize point cloud
        TARGET_Pc = np.zeros([len(P_ROI), 3])
        TARGET_Color = np.zeros([len(P_ROI), 3])
        r = 2
        for i in range(len(P_ROI)):
            print(i)
            p1 = P_ROI[i, :] + np.asarray([-r, -r])
            p2 = P_ROI[i, :] + np.asarray([r, r])
            x1 = p1[0]; y1 = p1[1]
            x2 = p2[0]; y2 = p2[1]
            idx_use = np.where((MTI_tex[:,0] >= x1) & (MTI_tex[:,0] <= x2) & (MTI_tex[:,1] >= y1) & (MTI_tex[:,1] <= y2))
            # print(len(idx_use[0]))
            if len(idx_use[0]) == 0:
                TARGET_Pc[i, :] = np.asarray([0,0,0])
                TARGET_Color[i, :] = np.asarray([0,0,0])
                continue
            elif len(idx_use[0]) == 1:
                TARGET_Pc[i, :] = np.asarray(MTI_vtx[idx_use, :])
                TARGET_Color[i, :] = np.asarray(colorvec[idx_use,:])
                continue
            tex_use = np.asarray(MTI_tex[idx_use, :])
            vtx_use = np.asarray(MTI_vtx[idx_use, :])
            color_use = np.asarray(colorvec[idx_use,:])
            vtx_use = np.asarray(vtx_use[0])
            color_use = np.asarray(color_use[0])
            TARGET_Pc[i,:] = np.mean(vtx_use, axis = 0)
            TARGET_Color[i,:] = np.asarray(np.mean(color_use, axis = 0), dtype = np.int32)

        # Viz the colorized point cloud results
        print(TARGET_Pc)
        print(TARGET_Color)
        obj_actor = BTL_VIZ.ActorNpyColor(TARGET_Pc, TARGET_Color)
        BTL_VIZ.VizActor([obj_actor])

        return TARGET_Pc, TARGET_Color, colorvec

    def GetTargetPc(self):

        # Get the colorized point cloud from texture and vertex data
        # Get the point cloud relative to the MTI frame
        P_ROI = self.BTL_Draw()

        # TARGET_Pc, TARGET_Color, colorvec = self.GetMtiTexColor(P_ROI)
        #
        # R_final = np.asarray([[-0.9995, 0.0167, 0.0253], \
        #                      [-0.0219, -0.9749, -0.2214], \
        #                       [0.0210, -0.2218, 0.9749]])
        # t_final = np.asarray([-2.7019, 5.7019, -0.2929])
        #
        # # Transfer to the MTI frame
        # TARGET = TARGET_Pc * 100
        # self.ST.plot_scan(TARGET)
        # N_final = len(TARGET)
        #
        # # print(TARGET)
        # np.save('test.npy', TARGET)
        # TARGET_mti = np.matmul(R_final, np.transpose(TARGET)) + np.transpose(np.tile(t_final, (N_final, 1)))
        # TARGET_mti = np.transpose(TARGET_mti)
        # # print(TARGET_mti[0, :])
        # TARGET_mti = TARGET_mti / 2.54  # cm to inch
        # print(TARGET[0,:])
        # print(TARGET_mti[0,:])
        # print(np.dot(R_final, np.transpose(TARGET)))
        # print(np.transpose(np.tile(t_final, (N_final, 1))))
        # print(TARGET_mti.shape)

        # Show the MTI data
        # self.ST.plot_scan(TARGET_mti)

        # raw_input("Please save the data as csv file")
        # np.savetxt("C:/Users/gm143/TumorCNC_brainlab/BTL/xyz_points.csv", TARGET_mti, delimiter=",", fmt = "%0.4f" )

    def MATLAB2Python(self):

        # Generate the MTI cutting region
        eng = matlab.engine.start_matlab()
        eng.cd('c:\Users\gm143\Documents\MATLAB\BTL')
        eng.main(nargout=0)

def CameraConfig():
    # return the camera configuration
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipe_profile = pipeline.start(config)
    return pipeline

def CheckboardMTI3D():
    # return the coordinates of the chessboard values
    # Based on the results of the grid points
    a = 1
    line_x = np.linspace(0.0, -1.0, num=10)
    line_y = np.linspace(0.0, -1.0, num=10)
    MOVE_X, MOVE_Y = np.meshgrid(line_x, line_y)
    Points_2d = np.zeros((len(MOVE_X[:].ravel()), 2))
    Points_2d[:,0] = MOVE_X.ravel()
    Points_2d[:,1] = MOVE_Y.ravel()

    print "The chessboard MTI MOVE_X is", MOVE_X
    print "The chessboard MTI MOVE_Y is", MOVE_Y
    print "The chessboard MTI points are ", Points_2d

    return MOVE_X.ravel(), MOVE_Y.ravel()

def Inch2Direction(unit_move, move_x, move_y):

    # move_x: inches on x axis
    # mvoe_y: inches on y axis
    # UserIn: inches on z axis

    # unit_move = 0.05    # how many steps to be moved
    steps_x = np.int( np.floor(move_x / unit_move) )
    steps_y = np.int( np.ceil(move_y / unit_move) )
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

def PixelToRegion(x_center_2d, y_center_2d):
    # input the pixel and return the region within the pixel centers
    # input: a set of pixel coordinates
    # output: a set of points related to the region -- list and numpy inside
    p_center = np.asarray([x_center_2d, y_center_2d])
    radius = 2
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
    # PC = BTL_FastPc.fastpc()

    # Get the color vector, vertex and textures
    # test.GetMtiTexColor()

    # Get the target point cloud -- with transformation
    # test.CaptureTemplateImg(L = 640, W = 480)
    # test.CaptureTemplateImg(L = 1280, W = 720)
    # PC.GetVerticesTexture()
    # test.GetTargetPc()

    # Draw the region
    test.BTL_Draw()

    # Calibration
    # for i in range(10):
    # MOVE_X, MOVE_Y = CheckboardMTI3D()
    # test.Scan3d(MOVE_X, MOVE_Y)

    # Ablation
    test.MATLAB2Python()
    # test.RegionCut_ANN()
    # test.CaptureTemplateImg(L = 640, W = 480)
    # test.CaptureTemplateImg(L = 1280, W = 720)
    # PC.GetVerticesTexture()

    # Scanning
    # test.ScanPoints()
    # test.GetCentroidFromImg()