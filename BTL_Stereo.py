import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
import sys
import os
sys.path.append(os.path.abspath('/home/maguangshen/PycharmProjects/BTL_GS/BTL/StereoVision/stereovision'))
import point_cloud
from multiprocessing import Pool
import glob
import BTL_DataConvert
import BTL_VIZ

class btl_stereo():

    def __init__(self):
        self.w = 640
        self.h = 480

    def Stereo_Calib(self):

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        objp = np.zeros((9 * 6, 3), np.float32)
        objp[:, : 2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 19  # add 19 mm to the grid

        # Arrays to store object points and image points from all images
        objpoints = []      # 3d points in real world space
        imgpointsR = []     # 2d points in image plane
        imgpointsL = []

        # Call all saved images
        for i in range(0, 30):  # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0

            t = str(i + 1)
            fname_left = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Left/' + t + '.jpg'
            fname_right = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Right/' + t + '.jpg'

            if (os.path.isfile(fname_left) == False):
                continue

            ChessImaL = cv2.imread(fname_left)  # Left side
            ChessImaR = cv2.imread(fname_right)  # Right side

            gray_left = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)

            retR, cornersR = cv2.findChessboardCorners(gray_right, (9, 6),
                                                       cv2.CALIB_CB_FAST_CHECK)  # Define the number of chees corners we are looking for
            retL, cornersL = cv2.findChessboardCorners(gray_left, (9, 6), cv2.CALIB_CB_FAST_CHECK)  # Left side

            if (True == retR) & (True == retL):
                objpoints.append(objp)
                cv2.cornerSubPix(gray_right, cornersR, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(gray_left, cornersL, (11, 11), (-1, -1), criteria)

                img_left = cv2.drawChessboardCorners(ChessImaL, (9, 6), cornersL, retL)
                img_right = cv2.drawChessboardCorners(ChessImaR, (9, 6), cornersR, retR)

                imgpointsR.append(cornersR)
                imgpointsL.append(cornersL)
                img_hstack = np.hstack([img_left, img_right])
                cv2.imshow('img', img_hstack)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        return objpoints, imgpointsL, imgpointsR, ChessImaL, ChessImaR

    def Stereo_Paras(self):

        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints, imgpointsL, imgpointsR, ChessImaL, ChessImaR = self.Stereo_Calib()
        gray_left = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)

        # Determine the new values for different parameters
        # Right camera calibration
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                                imgpointsR,
                                                                gray_right.shape[::-1], None, None)
        hR, wR = ChessImaR.shape[:2]    # The size of the image height: 720, width: 1280
        OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

        # ROI used to crop the image
        # Left camera calibration
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                                imgpointsL,
                                                               gray_left.shape[::-1], None, None)
        hL, wL= ChessImaL.shape[:2]
        OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

        # Check the reprojection errors
        mean_error_L = 0
        tot_error_L = 0
        mean_error_R = 0
        tot_error_R = 0

        for i in xrange(len(objpoints)):
            imgpoints2_L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            error = cv2.norm(imgpointsL[i], imgpoints2_L, cv2.NORM_L2) / len(imgpoints2_L)
            tot_error_L += error

        for i in xrange(len(objpoints)):
            imgpoints2_R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
            error = cv2.norm(imgpointsR[i], imgpoints2_R, cv2.NORM_L2) / len(imgpoints2_R)
            tot_error_R += error

        print("Left total error: ", tot_error_L)
        print("Left mean error: ", tot_error_L / len(objpoints))
        print("Right total error: ", tot_error_R)
        print("Right mean error: ", tot_error_R / len(objpoints))

        # Stereo calibrate function
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                  imgpointsL,
                                                                  imgpointsR,
                                                                  mtxL,
                                                                  distL,
                                                                  mtxR,
                                                                  distR,
                                                                  gray_right.shape[::-1],
                                                                  criteria_stereo,
                                                                  flags)

        print("The translation between the first and second camera is ", T)

        # StereoRectify function
        rectify_scale= 0 # if 0 image croped, if 1 image nor croped
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                          gray_right.shape[::-1], R, T,
                                                          rectify_scale, (0,0) )  # last paramater is alpha, if 0= croped, if 1= not croped

        print("The Q matrix is", Q)

        # initUndistortRectifyMap function -- map the images to the undistorted images
        Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                      gray_left.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
        Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                      gray_right.shape[::-1], cv2.CV_16SC2)

        return Left_Stereo_Map, Right_Stereo_Map, Q

    def Disparity_Map(self):

        # Filtering
        kernel = np.ones((3, 3), np.uint8)

        # Create StereoSGBM and prepare all parameters
        window_size = 3
        min_disp = 2
        num_disp = 130 - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            disp12MaxDiff = 5,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2)

        # Used for the filtered image
        stereoR = cv2.ximgproc.createRightMatcher(stereo)   # Create another stereo for right this time

        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 1.8
        visual_multiplier = 1.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        return stereo, stereoR, wls_filter

    def Stereo_Show(self):

        # Call the two cameras
        camL = cv2.VideoCapture(0)
        camR = cv2.VideoCapture(1)
        # Increase the resolution
        camL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        # usb bandwidth problem
        camL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        kernel = np.ones((3, 3), np.uint8)
        min_disp = 2
        num_disp = 130 - min_disp
        Left_Stereo_Map, Right_Stereo_Map, Q = self.Stereo_Paras()
        stereo, stereoR, wls_filter = self.Disparity_Map()

        while True:

            retR, frameR = camR.read()
            retL, frameL = camL.read()

            Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

            grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

            # Compute the 2 images for the Depth_image
            disp_use = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
            disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
            dispL = disp
            dispR = stereoR.compute(grayR, grayL)
            dispL = np.int16(dispL)
            dispR = np.int16(dispR)

            # Using the WLS filter
            filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            filteredImg = np.uint8(filteredImg)
            disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect
            closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

            # Colors map
            dispc = (closing - closing.min()) * 255
            dispC = dispc.astype(np.uint8)      # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
            disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
            filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

            # points = cv2.reprojectImageTo3D(disp_use, Q)
            # colors = Left_nice
            # pc = points.reshape(-1, 3)
            # pc = point_cloud.PointCloud(points, colors)
            # pc.write_ply('test.ply')

            cv2.imshow('Disparity', disp)
            cv2.imshow('Closing',closing)
            cv2.imshow('Color Depth',disp_Color)
            cv2.imshow('Filtered Color Depth', filt_Color)
            cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        camR.release()
        camL.release()
        cv2.destroyAllWindows()

    def Stereo_Test(self):

        # Call the two cameras
        camL = cv2.VideoCapture(0)
        camR = cv2.VideoCapture(1)
        # Increase the resolution
        camL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        # usb bandwidth problem
        camL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        kernel = np.ones((3, 3), np.uint8)
        min_disp = 2
        num_disp = 130 - min_disp
        Left_Stereo_Map, Right_Stereo_Map, Q = self.Stereo_Paras()
        stereo, stereoR, wls_filter = self.Disparity_Map()

        # Test with the results
        width = self.w
        height = self.h
        focal_length = 0.8 * width
        Q = np.float32([[1, 0, 0, -0.5 * width],
                        [0, -1, 0, 0.5 * height],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])

        # Generate a colorized point cloud
        retR, frameR = camR.read()
        retL, frameL = camL.read()
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                              0)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT, 0)

        # images_undistort = np.hstack([Left_nice, Right_nice])
        # cv2.imshow('Disparity', images_undistort)
        # cv2.waitKey()

        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp_use = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0

        disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        # Calculation allowing us to have 0 for the most distant object able to detect
        disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
        # Colors map
        dispc = (closing - closing.min()) * 255
        # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        dispC = dispc.astype(np.uint8)
        disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

        points = cv2.reprojectImageTo3D(dispC, Q)
        colors = Left_nice
        mask = disp_use > disp_use.min()
        out_points = points[mask]
        out_colors = colors[mask]
        pc = out_points.reshape(-1, 3)
        colors = out_colors.reshape(-1, 3)
        # vtk_pc = DataConvert.npy2vtk(pc)
        # BTL_VIZ.VizVtk([vtk_pc])
        pc_use = point_cloud.PointCloud(out_points, out_colors)
        pc_use.write_ply('pc_color.ply')
        actor_color = BTL_VIZ.ActorNpyColor(pc, colors)
        BTL_VIZ.VizActor([actor_color])

        camR.release()
        camL.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    test = btl_stereo()
    # test.Stereo_Paras()
    test.Stereo_Test()
