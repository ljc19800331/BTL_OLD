import numpy as np
from VideoCapture import Device
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import thread
from controlMovement import ControlMovement
from controlLaser import ControlLaser
from controlDAQ import ControlDAQ
from convertSteering import ConvertSteering
from scanTypes import ScanTypes
import BTL_PNP
from BTL_PNP import *

class webcamVideo:

    def __init__(self):

       print "Initializing video."

       self.cap = cv2.VideoCapture(1)

       # Config the realsense
       # self.pipeline = rs.pipeline()
       # self.config = rs.config()
       # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
       # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
       # # Start streaming
       # self.pipeline.start(self.config)

    def __del__(self):
       self.cap.release()

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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # cv2.waitKey(1)
        finally:
            # Stop streaming
            self.pipeline.stop()

    def videoFunction(self):

        # Capture and save the label image
        curr_frame = 0
        img = cv2.imread('C:\Users\gm143\TumorCNC_brainlab\BTL\color_test.png')
        while (curr_frame < 20):
            ret, frames = self.cap.read()
            img = frames
            if curr_frame == 19:
                cv2.imwrite('C:/Users/gm143/TumorCNC_brainlab/BTL/color_test.png', img)
                print "Save the standard image"
            curr_frame += 1

        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        print "The region is ", r

        try:
            while True:
                ret, frames = self.cap.read()
                # color_frame = frames.get_color_frame()
                # color_image = np.asanyarray(color_frame.get_data())
                color_image = frames
                ROI_image = color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                grey_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_image)
                Spot_center = (maxLoc[0] + np.int(r[0]), maxLoc[1] + np.int(r[1]))
                cv2.circle(color_image, Spot_center, 30, (0, 0, 0), 2)
                # grey_3d = np.dstack((grey_image, grey_image, grey_image))
                # images = np.hstack((color_image, grey_3d))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                cv2.waitKey(1)
        finally:
            # Stop streaming
            self.cap.release()
            cv2.destroyAllWindows()

    def startVideo(self):
       thread.start_new_thread(self.videoFunction, ())
       return

    def saveFrame(self,filename):
       ret, frame = self.cap.read()
       cv2.imwrite(filename, frame)

if __name__ == "__main__":

    test = PNP()
    web = webcamVideo()
    thread.start_new_thread(web.videoFunction, ())
    # thread.start_new_thread(web.RsCapture(), ())
    test.ScanPoints()
