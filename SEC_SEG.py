# This is the function for object segmentation
# Segment the background color
# Watershred segmentation
# Goal: Segment the surgical region
# Problem definition 1: The clear object boundary
# Problem definition 2: The background color is similar
# Problem definition 3: The noisy level is relatively low
# Problem definition 4: The surgical region is well defined
# Problem definition 5: A clear boundary
# Problem definition 6: The contour is always a rectangular (or a fixed shape)

# ref1: Multiple images in one window
# http://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
# ref2: Writing video with multiple windows
# https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
# ref3: Mask
# https://mmeysenburg.github.io/image-processing/03-drawing-bitwise/
# ref4: Realtime color detection
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
# ref5: computer vision and image processing expert
# https://www.pyimagesearch.com/pyimagesearch-gurus/

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

class sec_seg:

    def __init__(self):
        # imgpath = '/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/test_3.png'
        imgpath = []
        img = []

    def LoadImg(self):
        self.img = cv2.imread(self.imgpath)
        return self.img

    def ImgViz(self):
        plt.figure(1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))    # The imshow function is targeted for RGB images
        plt.show()

    def ImgThreshold(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        print(self.imgpath)
        for i in xrange(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def ImgAdaptThreshold(self):
        # Maybe different region has different lighting condition
        # The adaptive algorithm will calculate the threshold for a small region and we set the
        # each region as a separated region and get the result.
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img,5)                         # This step is important
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for i in xrange(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def ImgWaterThreshold(self):

        x, y = np.indices((80, 80))

        # gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #
        # plt.imshow(thresh, cmap = 'gray')
        #
        # # noise removal
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        #
        # # sure background area
        # sure_bg = cv2.dilate(opening,kernel,iterations=3)
        #
        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        #
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg,sure_fg)
        #
        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)
        #
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        #
        # # Now, mark the region of unknown with zero
        # markers[unknown==255] = 0
        #
        # markers = cv2.watershed(gray, markers)
        # gray[markers == -1] = [255,0,0]
        #
        # plt.imshow(sure_bg, cmap = 'gray')

    def ImgMask(self):
        # Ref: https://stackoverflow.com/questions/25074488/how-to-mask-an-image-using-numpy-opencv
        # Test the mask function
        img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/test_1.png')
        height, width, depth = img.shape
        circle_img = np.zeros((height, width), np.uint8)
        print (circle_img)
        cv2.circle(circle_img, (width/2, height/2), 280, 1, thickness = -1)
        print( circle_img)
        masked_data = cv2.bitwise_and(img, img, mask = circle_img)
        # The pixel coordinates of the
        cv2.imshow("masked", masked_data)
        cv2.waitKey(0)

    def ImgKmeanCluster(self):

        # Ref:http://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_segmentations.html
        # Ref:https://github.com/jayrambhia/superpixels-SLIC/blob/master/SLICcv.py

        # This method is too slow

        # example from scikit image

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

        # data = np.asanyarray(segments)
        # np.save('npydata' + '.npy', data)
        # Kmean cluster for color image using scikit image
        # Define a rectangular region and segment the region based on the color and edge
        # Given the image mask, output the region image
        # Core idea: The K mean cluster in the original method
        # Load the image
        # img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/test_1.png')
        # print(img)
        # RGB to LAB
        # img_lab = color.rgb2lab(img)
        # print(img_lab)

        # Superpixels
        # N_grid = 2000

        # Label the boundary

        # Mean values for each sub regions

        # Kmean clusters

class sec_obj:

    # This class is defined for object segmentation and object recognition
    # Goal: Find the rectangular object and ignore other distractors
    # input: BW of the edge result
    # output: The coarse region of the contour: rectangular by default

    def __init__(self):
        img = []

    def exp(self):

        while(1):
            img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/img_phantom_3.png')
            print(img.shape)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            print(img_gray.shape)
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            print(img_gray.shape)
            cv2.imshow("Gray", img_gray)
            edged = cv2.Canny(img_gray, 10, 250)
            cv2.imshow("Edge", edged)
            cv2.waitKey(5)
        # plt.figure(1)
        # plt.imshow(edged)  # The imshow function is targeted for RGB images
        # plt.show()

        # plt.imshow(img_gray)
        # plt.show()
        # # cv2.imshow("Gray", img_gray)
        # # cv2.waitKey(5)
        #
        # # Detect edges in the image

        # # cv2.imshow("Edged", edged)
        # plt.imshow(edged)
        # plt.show()
        # frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Res = cv2.Canny(frame, 100, 200)
        # plt.imshow(Res)
        # plt.show()

    def exp_video(self):

        # Ref: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

        # Create a VideoCapture object
        cap = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        while (True):
            ret, frame = cap.read()

            if ret == True:

                # Write the frame into the file 'output.avi'
                out.write(frame)

                # Display the resulting frame
                cv2.imshow('frame', frame)

                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

                # When everything done, release the video capture and video write objects
        cap.release()
        out.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def exp_Rec(self):

    # ref: https://pythonprogramming.net/canny-edge-detection-gradients-python-opencv-tutorial/
        cap = cv2.VideoCapture(0)

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

    def exp_RedRegion(self):

        # Realtime segmentation based on red color
        # Load the image
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        # frame = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/IMG_3781.JPG')
        img = frame

        # red color boundaries (R,B and G)
        lower = [1, 0, 20]
        upper = [60, 40, 200]

        # # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        ret, thresh = cv2.threshold(mask, 40, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(output, contours, -1, 255, 3)

            # find the biggest area
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            # draw the book contour (in green)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the images
        cv2.imshow("Result", np.hstack([img, output]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

    def exp_ConRec(self):

        # Find the object with the specific contour -- Rectangular
        # The problem assumes the problem to be 4 vertices
        # After region detection, threshold the content inside the region
        # Ref: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

        # Basic video setting
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        # Realtime Loop
        while (True):

            # Load the image
            _, img = cap.read()
            # img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/img_crop_1.png')

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
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                rec_img = np.zeros((height, width), np.uint8)
                if len(approx) == 4:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
                    X_approx, Y_approx = list2pt(approx)      # The parameters for the rectangle
                    mask = np.zeros(img.shape, dtype="uint8")
                    # Create the mask region as the rectangular
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

    def exp_Color(self):

        # Detect the region based on the color threshold -- Realtime
        # ref: https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

        # Construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", help="path to the image")
        # args = vars(ap.parse_args())
        # # load the image
        # image = cv2.imread(args["image"])

        # Load the realtime image
        cap = cv2.VideoCapture(0)

        # Define the list of color boundaries
        boundaries = [([17, 15, 100], [50, 56, 200]), \
                      ([86, 31, 4], [220, 88, 50]), \
                      ([25, 146, 190], [62, 174, 250]), \
                      ([103, 86, 65], [145, 133, 128])]

        red_color = boundaries[0]
        while(True):
            _, img = cap.read()
            lower = red_color[0]; upper = red_color[1]
            lower = np.array(lower, dtype= "uint8" )
            upper = np.array(upper, dtype= "uint8" )
            # Find the colors within the specified boundaries and apply the mask
            mask = cv2.inRange(img, lower, upper)
            output = cv2.bitwise_and(img, img, mask=mask)
            # Show the images
            cv2.imshow("images", np.hstack([img, output]))
            cv2.waitKey(5)

def list2pt(obj):
    # input: specific list
    # output: min(x), max(x), min(y), max(y)
    X = np.asarray( (obj[0][0][0], obj[1][0][0], obj[2][0][0], obj[3][0][0]))
    Y = np.asarray((obj[0][0][1], obj[1][0][1], obj[2][0][1], obj[3][0][1]))
    p1 = (obj[0][0][0], obj[0][0][1])
    p2 = (obj[1][0][0], obj[1][0][1])
    p3 = (obj[2][0][0], obj[2][0][1])
    p4 = (obj[3][0][0], obj[3][0][1])
    return X, Y

if __name__ == '__main__':

    test = sec_obj()
    test.exp_Color()

    # test = sec_seg()
    # test.ImgKmeanCluster()

    # test = sec_seg()
    # test.ImgMask()

    # test = sec_obj()
    # test.exp_video()
    # test.exp_ConRec()

    # test = sec_obj()
    # test.exp_video()

    # Test exp_rec
    # test = sec_obj()
    # test.exp_Rec()
    # test.exp_RedRegion()

    # Test load image, viz image
    # imgpath = '/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/img_crop_1.png'
    # test = sec_seg()
    # test.imgpath = imgpath
    # img = test.LoadImg()
    # test.ImgViz()

    # Test threshold method
    # test.ImgThreshold()           # Can change the threshold to detect the region

    # Test adaptive thresholding method
    # test.ImgAdaptThreshold()

    # Test the Watershred threshold method
    # test.ImgWaterThreshold()