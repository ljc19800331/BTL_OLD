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

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        plt.imshow(thresh, cmap = 'gray')

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(gray, markers)
        gray[markers == -1] = [255,0,0]

        plt.imshow(sure_bg, cmap = 'gray')

    def ImgKmeanCluster(self):
        # Kmean cluster for color image
        # Define a rectangular region and segment the region based on the color and edge
        # Using skimage
        a = 1


class sec_obj:

    # This class is defined for object segmentation and object recognition
    # Goal: Find the retangular object and ignore other distractors
    # input: BW of the edge result
    # output: The coarse region of the contour: retangular by default

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

        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        while (True):
            _, img = cap.read()
            # Load the image
            # img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/img_crop_1.png')
            img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            # print(img.shape)
            # cv2.imshow("Small", img_small)
            # plt.subplot(2, 2, 1), plt.imshow(img_gray, 'gray')
            # plt.show()
            # cv2.imshow("Gray", img_gray)

            # Detect edges in the image
            H_show = cv2.hconcat((img_gray, img_gray))
            V_show = cv2.vconcat((H_show, H_show))
            edged = cv2.Canny(img_gray, 10, 250)
            final_frame = cv2.hconcat((img_gray, edged))
            cv2.imshow('lena', V_show)

            # cv2.imshow("Edged", edged)

            out.write(img)

            # construct and apply a closing kernel to 'close' gaps between 'white' pixels -- close the region
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            #cv2.imshow("Closed", closed)

            # find contours (i.e. the 'outlines') in the image and initialize the total number of regions found
            _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            total = 0

            # Find the contour with four vertices
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
                    print (approx)
                total += 1  # The number of target contour

            # Design a mask based no the four vertices
            Vtx = approx    # The coordinates of the four vertices

            # Mask the region and segment the region

            # threhold the pink color with a stable method

            # display the output
            #cv2.imshow("Output", img)
            cv2.waitKey(5)

        cv2.destroyAllWindows()
        cap.release()
        out.release()

if __name__ == '__main__':

    test = sec_obj()
    # test.exp_video()
    test.exp_ConRec()

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
    # test.ImgThreshold()

    # Test adaptive thresholding method
    # test.ImgAdaptThreshold()

    # Test the Watershred threshold method
    # test.ImgWaterThreshold()
