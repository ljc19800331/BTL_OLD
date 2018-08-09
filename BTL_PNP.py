# Calibration PNP
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
        flag = raw_input("if viz data? (yes or no)")
        if flag == 'no':
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(self.config)

    def SaveRealtimeImg(self):
        a = 1
        # save the image when the laser spot is moving
        # perform the laser scanning process
        # thread.start_new_thread(self.RsCapture(), ())
        # self.ScanPoints()
        t1 = Thread(target=self.VideoCapture())
        t2 = Thread(target = timer, args=("Timer2", 2, 5))
        t1.start()
        t2.start()

    def LaserDetect(self):
        # Detect the centroid of the laser spot
        # Read the image
        # img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png', 0)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Define the ROI manually
        # https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
        img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The region is ", r

        # Detect the region
        # https://www.youtube.com/watch?v=NtJXi7u_fJo
        flag = raw_input("realtime or static?")

        if flag == "realtime":
            try:
                while True:
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                    grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
                    Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
                    cv2.circle(color_image, Spot_center, 30, (0,0,0), 2)
                    # grey_3d = np.dstack((grey_image, grey_image, grey_image))
                    # images = np.hstack((color_image, grey_3d))
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', color_image)
                    cv2.waitKey(1)
            finally:
                # Stop streaming
                self.pipeline.stop()

        if flag == "static":
            color_image = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
            ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
            Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
            cv2.circle(color_image, Spot_center, 30, (0, 0, 0), 2)
            print "The Spot center is ", Spot_center
            # grey_3d = np.dstack((grey_image, grey_image, grey_image))
            # images = np.hstack((color_image, grey_3d))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(0)

    def CamIntrinsic(self):

        # intrinsic parameters of the which imager?
        # https://github.com/IntelRealSense/librealsense/issues/869
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Intrinsics & Extrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            print(color_intrin)

            curr_frame = 0
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                # Intrinsics & Extrinsics
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
                    color_frame.profile)
                #print(depth_intrin)
                #print(color_intrin)

                if (curr_frame == 30):
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imwrite('C:/Users/gm143/TumorCNC_brainlab/BTL/color_test.png', color_image)
                curr_frame += 1

        finally:
            # Stop streaming
            self.pipeline.stop()
        return color_intrin

    def RsCapture(self):
        # video capture for pyrealsense
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # if not depth_frame or not color_frame:
                # continue

                # Convert images to numpy arrays
                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                # images = np.hstack((color_image, depth_colormap))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                cv2.waitKey(1)
        finally:
            # Stop streaming
            self.pipeline.stop()

    def VideoCapture(self):

        while(True):

            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def MoveOrigin(self):
        # move to the origin
        x = 0
        y = 0
        self.CD.point_move(x, y, 0.001)

    def MovePoint(self, move_x, move_y):
        # move a point to a specific position (x,y)
        # move_x: move x value from the origin
        # move_y: move y value from the origin

        # update the original .npy files
        filename = 'ScanAngles.npy'
        if os.path.exists(filename):
            os.remove(filename)

        print 'Set center of scan.'
        x_c = 0; y_c = 0                # The starting points in x and y

        # move_x = 1  # inches
        # move_y = -1  # inches
        UserIn = self.Inch2Angle(move_x, move_y)
        centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn)

    def ScanPoints(self):

        filename = 'ScanAngles.npy'
        if os.path.exists(filename):
            os.remove(filename)

        mtiScanSpan = [1.6, 1.6]        # The scanning region of the MTI
        pointDistance = 0.1           # inches between each point on the scan

        # Move the center of the scan using wasd
        print 'Set center of scan.'
        x_c = 0; y_c = 0                # The starting angles value -- origin

        # Define the initial position from the origin -- different from (0,0)
        UserIn = self.Inch2Angle(1.6, -1.6)

        centerScan, z_dist, center_angles = self.CM.controlPoint_v1([x_c], [y_c], UserIn)
        # centerScan, z_dist, center_angles = self.CM.controlPoint([x_c], [y_c])
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
        np.save('C:/Users/gm143/TumorCNC_brainlab/BTL/P_3D.npy', target_points)
        # find the origin and the coordinate system
        # max, min -- x,y mean z -- checking the values

    def PointSTA(self, Points):

        # Statistical for the points
        # Points = np.random.randint(5, size=(10, 3))
        # How to threshold the boundary and the data?

        print("The points are ", Points)
        # min and max z
        z_min = np.min(Points[:, 2])
        z_max = np.max(Points[:, 2])
        diff_zminmax = z_max - z_min
        # print("z_mim is ", z_min, "z_max is ", z_max, "diff_zminmax is ", diff_zminmax)
        print "The different of z min and max is ", diff_zminmax

        # mean interval for x
        x = Points[:,0]
        x_1 = x[:-1]
        x_2 = x[1:]
        x_int = np.absolute(x_2 - x_1)
        x_int_mean = np.mean(x_int)
        # print "The x_1 is ", x_1, "The x_2 is ", x_2, "The x_int is ", x_int, "The mean x_int is ", x_int_mean
        print "The mean interval of x is ", x_int_mean

        # mean interval for y
        y = Points[:,1]
        y_1 = y[:-1]
        y_2 = y[1:]
        y_int = np.absolute(y_2 - y_1)
        y_int_mean = np.mean(y_int)
        # print "The y_1 is ", y_1, "The y_2 is ", y_2, "The y_int is ", y_int, "The mean y_int is ", y_int_mean
        print "The mean interval of x is ", y_int_mean

    def Inch2Angle(self, move_x, move_y):
        # move_x: inches on x axis
        # mvoe_y: inches on y axis
        # UserIn: inches on z axis
        # move_x = 1  # inches
        # move_y = -1  # inches
        unit_mvoe = self.arrowMove      # how long for each step
        steps_x = move_x / unit_mvoe
        steps_y = move_y / unit_mvoe
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

    def Extract2DPoints(self):
        # Extract the 2D coordinates from the images
        path, dirs, files = next(os.walk("C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser"))

        img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser/P_2.png')
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The region is ", r

        P_2D = np.zeros([  len(range(1,len(files))), 2 ])
        idx = 0
        for i in range(2, len(files) + 1):
            img_name = 'C:/Users/gm143/TumorCNC_brainlab/Data/ImgLaser/' + 'P_' + str(i) + '.png'
            img = cv2.imread(img_name)
            ROI_image = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
            Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
            P_2D[idx,:] = Spot_center
            idx += 1
        print(P_2D)
        np.save('C:\Users\gm143\TumorCNC_brainlab\BTL\P_2D.npy', P_2D)

    def pnp(self):

        # pnp operation to calculate the result
        # Intrinsic parameters
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Intrinsics & Extrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print "The color_intrin is ", color_intrin

        # The camera matrix
        # width: 640, height: 480, ppx: 312.14, ppy: 238.018, fx: 619.846, fy: 619.846, model: None
        # im = cv2.imread("C:\Users\gm143\TumorCNC_brainlab\Data\ImgLaser\P_2.png")
        camera_matrix = np.array(
            [[619.846, 0, 312.14],
             [0, 619.846, 238.018],
             [0, 0, 1]], dtype="double"
        )
        print(camera_matrix)

        # Read the 2D points
        P_2D = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_2D.npy')
        print(P_2D.shape)

        # Read the 3D points
        p3D = np.load('C:\Users\gm143\TumorCNC_brainlab\BTL\P_3D.npy')
        P_3D = p3D[1:]
        print(P_3D)

        # normalize the 3D points to the origin of the first point
        P_3D_new = P_3D - np.tile(P_3D[0,:], (270, 1))
        print(P_3D_new)

        # PNP to find the relation and transformation
        # Assume no distortion
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(P_3D_new, P_2D, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat = cv2.Rodrigues(rotation_vector)[0]
        print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Translation Vector:\n {0}".format(translation_vector)
        print "The rotation matrix is ", rmat

        # Test the 3D point
        test_3D = P_3D_new  # after normalization
        test_2D = P_2D

        print "The test_3D is ", test_3D
        print "The test 2D is ", test_2D

        (test_point2D, jacobian) = cv2.projectPoints(np.array([test_3D]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Reshape the test_point2D (L, L, L) to (L, L)
        test_point2D = np.reshape(test_point2D, (len(test_point2D), 2))

        print("The test 2D point is ", test_point2D)
        print("The shape of test 2D point is ", test_point2D.shape)

        # print "The Jacobian is ", jacobian
        # img = cv2.imread("C:\Users\gm143\TumorCNC_brainlab\Data\ImgLaser\P_1.png")
        # cv2.circle(img, (np.int(test_point2D[0][0][0]), np.int(test_point2D[0][0][1])), 30, (0,0,0), 2)
        # cv2.circle(img, (np.int(test_point2D[0][0][0]), np.int(test_point2D[0][0][1])), 5, (0,0,0), 2)
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)

        # Calculate the average of the error (labelled as pixel values)
        diff = test_2D - test_point2D
        mTRE = (np.sum(np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1])))) / len(test_point2D)
        print("The diff of 2D image points are", diff.shape)
        # mean target registration error
        print("The mTRE (2D image) is ", mTRE)

        return rmat, translation_vector, mTRE

    def VizPNP(self):

        # Viz the two coordinate system
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.SetBackground(.1, .2, .3)  # Background dark blue

        transform_1 = vtk.vtkTransform()
        transform_1.Translate(0.0, 0.0, 0.0)
        transform_1.RotateX(45)

        transform_2 = vtk.vtkTransform()
        transform_2.Translate(1.0, 1.0, 1.0)

        # Design the first axes
        axes_1 = vtk.vtkAxesActor()
        axes_1.SetUserTransform(transform_1)
        renderer.AddActor(axes_1)

        axes_2 = vtk.vtkAxesActor()
        axes_2.SetUserTransform(transform_2)
        renderer.AddActor(axes_2)

        renderer.ResetCamera()
        renderWindow.Render()
        # begin mouse interaction
        renderWindowInteractor.Start()


def timer(name, delay, repeat):
    print "Timer: " + name + " Started"
    while repeat > 0:
        time.sleep(delay)
        print name + ": " + str(time.ctime(time.time()))
        repeat -= 1
    print "Timer: " + name + " Completed"

if __name__ == "__main__":

    test = PNP()
    test.VizPNP()

    # intrin = test.CamIntrinsic()
    # rmat, tvec, mTRE = test.pnp()
    # test.Extract2DPoints()
    # test.SaveRealtimeImg()
    # test.LaserDetect()
    # test.CamIntrinsic()
    # test.VideoCapture()
    # test.RsCapture()
    # test.PointSTA()
    # test.MoveOrigin()
    # test.ScanPoints()
    # test.MovePoint(1.6, -1.6)
    # move_x = 1
    # move_y = 1
    # test.Inch2Angle(move_x, move_y)