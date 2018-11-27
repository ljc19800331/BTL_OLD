# 2D calibration and other applications
# Calibration and others
# Pose estimation
import cv2
import numpy as np
import glob
import click
import time
import point_cloud

class BTL5D():

    def __init__(self):
        a = 1
        self.w = 640
        self.h = 480
        # self.cap = cv2.VideoCapture(1)
        # self.left = cv2.VideoCapture(0)
        # self.right = cv2.VideoCapture(1)
    def CamRealtime(self):
        # Capture the 2D image in realtime
        try:
            while True:
                ret, frames = self.cap.read()
                # color_frame = frames.get_color_frame()
                # color_image = np.asanyarray(color_frame.get_data())
                color_image = frames
                # ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                # grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
                # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
                # Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
                # cv2.circle(color_image, Spot_center, 30, (0, 0, 0), 2)
                # grey_3d = np.dstack((grey_image, grey_image, grey_image))
                # images = np.hstack((color_image, grey_3d))
                cv2.namedWindow('CAM', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('CAM', color_image)
                cv2.waitKey(1)
        finally:
            # Stop streaming
            self.cap.release()
            cv2.destroyAllWindows()
    def CaptureImg(self):

        # Capture the images for monocular image
        Path = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/CaptureImg/'

        # Define the resolution
        CAMERA_WIDTH = self.w
        CAMERA_HEIGHT = self.h

        left = cv2.VideoCapture(0)
        right = cv2.VideoCapture(1)

        # Increase the resolution
        left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # save the left image
        _, Frame = left.read()
        cv2.imwrite('test_left.jpg', Frame)
        _, Frame = right.read()
        cv2.imwrite('test_right.jpg', Frame)

        left.release()
        right.release()
        cv2.destroyAllWindows()
    def CamCalibrate(self):

        # termination criteria
        CHESSBOARD_SIZE = (9, 6)
        CHESSBOARD_OPTIONS = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                              cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        print "Chessboard option is ", CHESSBOARD_OPTIONS

        OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
                                     np.float32)
        OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                   0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        print "The Object poitn zero is", OBJECT_POINT_ZERO
        OPTIMIZE_ALPHA = 0.25
        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                                0.001)
        # MAX_IMAGES = 64
        # gridsize = 20 # mm for the grid size
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        print("The coordinates of object points are ", objp)

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # Read the images and
        images = glob.glob('/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/CaptureImg/*.jpg')
        # print "images are ", images

        # frameID = 1
        # Path = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/CaptureImg_exp'
        # while(frameID < 17):
        for fname in images:
            # Read the images
            # img = cv2.imread(Path + str(frameID) + '.jpg')
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('img', gray)
            # cv2.waitKey(10)

            # Find the chess board corners
            # ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            hasCorners, corners = cv2.findChessboardCorners(gray,
                                                            CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
            # print("The ret is ", hasCorners)
            # print("The corners are", corners)

            if hasCorners == True:

                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners2, hasCorners)
                cv2.imshow('img', img)
                cv2.waitKey(500)

            # frameID += 1
        cv2.destroyAllWindows()

        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print "The ret is ", ret
        print "The mtx is ", mtx
        print "The dist is ", dist
        print "The rvecs is ", rvecs
        print "The tvecs is ", tvecs

        # Measure the reprojection error
        total_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print "total error: ", total_error / len(objpoints)

        return ret, mtx, dist, rvecs, tvecs
    def draw3DAxis(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img
    def draw3DCube(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img
    def PoseEstimation(self):

        # Obtain the camera matrix
        ret, mtx, dist, rvecs, tvecs = self.CamCalibrate()

        # PoseEstimation
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # For 3D Axis
        # axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        # For 3D Cube
        axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

        # Draw the 3D axis
        for fname in glob.glob('/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/CaptureImg/*.jpg'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vector
                # (rvecs, tvecs, inliers) = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                (success, rotation_vector, translation_vector) = cv2.solvePnP(objp, corners2, mtx, dist)
                # print("The retval is ", retval)
                print("The rvec is ", rotation_vector)
                print("The tvecs is ", translation_vector)
                # print("The inlieres is ", inliers)

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)
                img = self.draw3DCube(img, corners2, imgpts)
                cv2.imshow('img', img)
                k = cv2.waitKey(0) & 0xff
                # if k == 's':
                #     cv2.imwrite(fname[:6] + '.png', img)
        cv2.destroyAllWindows()
    def Project2DTo3D(self):

        # Project 2D points from image to the 3D world coordinates
        # Get the rotation and translation vectors
        ret, mtx, dist, rvecs, tvecs = self.CamCalibrate()

        # rmat = cv2.Rodrigues(rotation_vector)[0]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Find the translation and rotation
        img_test = cv2.imread("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/CaptureImg/1.jpg")
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_test, (9, 6), None)
        corners2 = cv2.cornerSubPix(gray_test, corners, (11, 11), (-1, -1), criteria)

        print "The corners are ", corners

        (success, rotation_vector, translation_vector) = cv2.solvePnP(objp, corners2, mtx, dist)
        rmat = cv2.Rodrigues(rotation_vector)[0]
        print "The rotation matrix is ", rmat
        print "The translation vector is ", translation_vector

        # The test 2D and 3D points
        cam_matrix = mtx                 # camera matrix (intrinsic matrix)
        trans_vec = translation_vector      # translation matrix
        rot_mat = rmat           # rotation matrix
        dist_mat = dist                     # distortion matrix

        pts2D = np.asarray([[437], [27.5], [1]])
        pts2D_InInv = np.matmul(np.linalg.inv(cam_matrix), pts2D)
        pts2D_InInv_Tvec = pts2D_InInv - trans_vec
        pts2D_InInv_Tvec_Rvec = np.matmul(np.linalg.inv(rot_mat), pts2D_InInv_Tvec)

        print("The camera matrix is ", cam_matrix)
        print("The pts2D_InInv is ", pts2D_InInv)
        print("The pts2D_InInv_Tvec is ", pts2D_InInv_Tvec)
        print("The pts2D_InInv_Tvec_Rvec is ", pts2D_InInv_Tvec_Rvec)
    def StereoShow(self):

        # Show the stereo images in realtime
        # a = 1
        # left = cv2.VideoCapture(0)
        # right = cv2.VideoCapture(2)
        # print (cv2.VideoCapture)

        # https://stackoverflow.com/questions/40236826/opencv-cheap-stereo-camera-cant-load-both-streams-at-once
        left = cv2.VideoCapture(0)
        right = cv2.VideoCapture(1)

        # Increase the resolution
        left.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        right.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        # usb bandwidth problem
        left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        try:
            while(True):
                ret, left_frame = left.read()
                ret, right_frame = right.read()
                # print "left shape", left_frame.shape
                # print "right shape", right_frame.shape
                # images = np.hstack((left_frame, right_frame))
                # cv2.namedWindow('CAM', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('left frame', left_frame)
                cv2.imshow('right frame', right_frame)
                # cv2.imwrite('test.jpg', left_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.waitKey(1)
        finally:
            left.release()
            right.release()
    def StereoCapImg(self):

        # Capture the images from
        # left = cv2.VideoCapture(0)
        # right = cv2.VideoCapture(2)
        # print (cv2.VideoCapture)

        curr_frame = 1
        left_dir = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Left/'
        right_dir = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Right/'

        while(curr_frame < 21):
            # raw_input("press enter to continue")
            # click.pause()

            # https://stackoverflow.com/questions/40236826/opencv-cheap-stereo-camera-cant-load-both-streams-at-once
            left = cv2.VideoCapture(0)
            right = cv2.VideoCapture(1)

            # Increase the resolution
            left.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            right.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

            # left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
            # usb bandwidth problem
            left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            time.sleep(1.0)
            ret, left_frame = left.read()
            ret, right_frame = right.read()

            left_name = left_dir + str(curr_frame) + '.jpg'
            right_name = right_dir + str(curr_frame) + '.jpg'

            cv2.imwrite(left_name, left_frame)
            cv2.imwrite(right_name, right_frame)

            cv2.imshow('left frame', left_frame)
            cv2.imshow('right frame', right_frame)

            cv2.waitKey(500)
            cv2.destroyAllWindows()
            left.release()
            right.release()
            curr_frame += 1
    def StereoCalibrate_exp(self):
        # use the example code for this problem
        CHESSBOARD_SIZE = (9, 6)
        CHESSBOARD_OPTIONS = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                              cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
                                     np.float32)
        OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                   0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        OPTIMIZE_ALPHA = 0.25
        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                                0.001)
    def StereoCalibrate(self, dirpath):

        # Calibrate the stereo camera
        CHESSBOARD_SIZE = (9, 6)        # Remember to change the grid size
        CHESSBOARD_OPTIONS = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                              cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        # print "Test the option", CHESSBOARD_OPTIONS
        OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
                                     np.float32)
        OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                   0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        OPTIMIZE_ALPHA = 0.25   # optimization parameter
        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                                0.001)

        # prepare object points
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # Read the images from the directory
        images = glob.glob(dirpath + '*.jpg')

        for fname in images:
            # Read the images
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            hasCorners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
            if hasCorners == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners2, hasCorners)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

        # Calibration
        imageSize = gray.shape[::-1]    # image size
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Measure the reprojection error
        total_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(objpoints)
        print "total error: ", mean_error

        return ret, mtx, dist, rvecs, tvecs, mean_error, objpoints, imgpoints, imageSize
    def StereoRec(self):

        # Stereo reconstruction following the procedure
        # Define the calibration images
        leftpath = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Left/'
        rightpath = '/home/maguangshen/PycharmProjects/BTL_GS/BTL/Right/'

        # Get the calibration coeffcients
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left, mean_error_left, \
        objpoints_left, imgpoints_left, imageSize_left = self.StereoCalibrate(leftpath)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right, mean_error_right, \
        objpoints_right, imgpoints_right, imageSize_left = self.StereoCalibrate(rightpath)

        print("Calibrating cameras together...")
        objectPoints = objpoints_left       # This is a list object
        imageSize = imageSize_left          # This is a size object
        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        print "The shape of obj points is ", objectPoints, type(objectPoints)
        print "The image size is ", imageSize

        (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
            objectPoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            imageSize, None, None, None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

        # Define the optimization alpha
        OPTIMIZE_ALPHA = 0.25

        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            imageSize_left, rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

        # print("Saving calibration...")
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, leftRectification,
            leftProjection, imageSize, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, rightRectification,
            rightProjection, imageSize, cv2.CV_32FC1)

        # linear interpolation
        REMAP_INTERPOLATION = cv2.INTER_LINEAR
        DEPTH_VISUALIZATION_SCALE = 2048

        CAMERA_WIDTH = self.w
        CAMERA_HEIGHT = self.h

        left = cv2.VideoCapture(0)
        right = cv2.VideoCapture(1)

        # Increase the resolution
        left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        stereoMatcher = cv2.StereoBM_create()
        stereoMatcher.setMinDisparity(4)
        stereoMatcher.setNumDisparities(128)
        stereoMatcher.setBlockSize(21)
        stereoMatcher.setROI1(leftROI)
        stereoMatcher.setROI2(rightROI)
        stereoMatcher.setSpeckleRange(16)
        stereoMatcher.setSpeckleWindowSize(45)

        leftFrame = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/BTL/test_left.jpg')
        rightFrame = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/BTL/test_right.jpg')

        fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
        fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        #
        depth = stereoMatcher.compute(grayLeft, grayRight)
        #     print "The depth is ", depth
        #
        #     cv2.imshow('left', leftFrame)
        #     cv2.imshow('right', rightFrame)
        #     # cv2.waitKey(0)

        print(depth)

        Q = dispartityToDepthMap

        print(Q)

        points = cv2.reprojectImageTo3D(depth, Q)
        colors = leftFrame

        pc = point_cloud.PointCloud(points, colors)
        pc.write_ply('pc_color.ply')

        #     print("The shape of the points are ", points.shape)
        #     # cv2.imshow('depth', depth * 3)
        #     # cv2.waitKey(0)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        left.release()
        right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    test = BTL5D()
    # test.CaptureImg()
    # test.CamRealtime()
    # test.StereoCapImg()
    # test.StereoViz()
    test.StereoShow()
    # test.StereoRec()
    # ret, mtx, dist, rvecs, tvecs = test.CamCalibrate()
    # test.PoseEstimation()
    # test.Project2DTo3D()