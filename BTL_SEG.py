# This code is used for 2D segmentation and visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import color
from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import time
import argparse
import DataConvert
from DataConvert import *

def KmeanClusterSeg():

    # low-level image segmentation methods -- example
    # Ref:http://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_segmentations.html
    # Ref:https://github.com/jayrambhia/superpixels-SLIC/blob/master/SLICcv.py

    # load image
    img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/others/superpixels-SLIC/test_1.png')

    # SLIC superpixel region segmentations
    time_start = time.clock()
    segments = slic(img, n_segments=2000, compactness=10, convert2lab='True')

    # boundary visualization based on the above result
    img_boundary = mark_boundaries(img, segments)
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)

    # Convert superpixels to the cell array
    print(segments)

    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.subplot(2, 2, 2), plt.imshow(segments)
    plt.subplot(2, 2, 3), plt.imshow(img_boundary)
    plt.show()

def MaskCircle():

    # Ref: https://stackoverflow.com/questions/25074488/how-to-mask-an-image-using-numpy-opencv
    # Test the mask function
    img = cv2.imread('/home/maguangshen/PycharmProjects/realsense/color.png')
    height, width, depth = img.shape
    circle_img = np.zeros((height, width), np.uint8)

    # Design the circle
    cv2.circle(circle_img, (width/2, height/2), 280, 1, thickness = -1)
    masked_data = cv2.bitwise_and(img, img, mask = circle_img)

    # Viz the mask and the resulted image
    cv2.imshow("masked", masked_data)
    cv2.waitKey(0)

def SegColor(cap):

    # Segment the region based on the color difference
    # Load the world frame
    # cap = cv2.VideoCapture(1)

    # Define the list of color boundaries
    boundaries = [([160, 100, 100], [255, 255, 255]),   \
                  ([86, 31, 4], [220, 88, 50]),     \
                  ([25, 146, 190], [62, 174, 250]), \
                  ([103, 86, 65], [145, 133, 128])]

    # Define the red color boundary
    red_color = boundaries[0]

    # Begin the control flow
    while (True):

        # read the single frame
        _, img = cap.read()

        # detect the color factor
        lower = red_color[0]
        upper = red_color[1]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # Show the images
        cv2.imshow("images", np.hstack([img, output]))
        cv2.waitKey(5)

def exp_ConRec():

    # Find the object with the specific contour -- Rectangular
    # The problem assumes the problem to be 4 vertices
    # After region detection, threshold the content inside the region
    # Ref: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

    # Basic video setting
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # The format of the video frame
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    # Realtime Loop
    while (True):

        # Load the image
        _, img = cap.read()

        # Convert the image to grayscale image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Detect edges in the image
        edged = cv2.Canny(img_gray, 10, 250)
        final_frame = cv2.hconcat((img_gray, edged))
        H_show_1 = cv2.hconcat((img_gray, edged))
        V_show = cv2.vconcat((H_show_1, H_show_1))

        # Record the video with each frame
        # out.write(img)

        # Construct and apply a closing kernel to 'close' gaps between 'white' pixels -- close the region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find contours (i.e. the 'outlines') in the image and initialize the total number of regions found
        _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        total = 0

        # Find the contour with four vertices
        height, width, depth = img.shape    # The parameters for the mask

        # Find all the contour that has four vertices
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            rec_img = np.zeros((height, width), np.uint8)

            # If the contour has 4 vertices, i.e the rectangle region
            if len(approx) == 4:

                # draw contour
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
                X_approx, Y_approx = list2pt(approx)          # The parameters for the rectangle

                # Create the mask region as the rectangular
                mask = np.zeros(img.shape, dtype="uint8")
                cv2.rectangle(mask, (min(X_approx), max(Y_approx)), (max(X_approx), min(Y_approx)), (255, 255, 255), -1)
                maskedImg = cv2.bitwise_and(img, mask)
                cv2.imshow("Masked Image", maskedImg)
                cv2.waitKey(5)

            total += 1  # The number of target contour

    # Post processing
    cv2.waitKey(5)
    cv2.destroyAllWindows()
    cap.release()
    # out.release()

def CannyEdge(cap):

    # Canny Edge Detection and Gradients
    # ref: https://pythonprogramming.net/canny-edge-detection-gradients-python-opencv-tutorial/
    # cap = cv2.VideoCapture(0)

    while (1):

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Original', frame)
        edges = cv2.Canny(frame, 100, 200)
        cv2.imshow('Edges', edges)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def SegColorReal():

    # Realtime segmentation based on red color
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

    # Load the image
    cap = cv2.VideoCapture(0)

    while(1):

        # Take each frame
        _, frame = cap.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower = np.array([0, 0, 0])
        upper = np.array([60, 40, 200])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # viz the resulted image
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':

    cap = cv2.VideoCapture(1)

    # Test the color
    # SegColor(cap)

    # Test the ImgKmeanCluster algorithm
    # KmeanClusterSeg()

    # Test MaskSeg
    MaskCircle()

    # Test CannyEdge
    # CannyEdge(cap)

    # Test Realtime red color segmentation
    # SegColorReal()

    # Test the Contour detection method
    # exp_ConRec()