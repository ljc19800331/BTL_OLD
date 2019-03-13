# This is a reconstruction algorithm for vessel segmentation
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
import cv2
import open3d

'''
The code structure
1. Load all the data files -- From DICOM or others
2. Seed -- output (x,y)
3. Contour -- output a series of (xn, yn) -- N is the number of output coordinates

The first simulation experiment
Goal: Create the 3D model of the vessel from the ultrasound image
1. input the ultrasound image (from single one image)
2. Manual choose the seed -- output the centroid of the trajectory -- show the trajectory
3. Manual create the contour -- output the 2D coordinates + the Z coordinates
4. A simple method of binarizing the region of the vessel and create a 3D model
5. A general platform for 3D reconstruction

LOOP:
1. read the image
2. detect the centroid
3. detect the contour
4. assign the Z axis coordinate

Create the 3D point cloud -- In the future it would be more difficult

reference:
1. https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

'''

class VS_track():

    def __init__(self):
        self.img_path = []
        self. a = 1

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def resample(image, scan, new_spacing=[1, 1, 1]):

    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)  ))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def VS_seed(img):

    # Input an image and output the seed coordinate
    # This is the main algorithm
    pos_seed = 1
    a = 1

    return pos_seed

def VS_contour(img):

    # input an image and output the vessel contour
    # This is the main algorithm
    pos_contour = 1

    return pos_contour

def VS_draw(img):

    global ix, iy, drawing, mode, P_circle
    drawing = False
    mode = True
    ix, iy = -1, -1
    P_ROI = []
    P_circle = np.zeros((1,2))

    # Mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode, P_circle

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:

            if drawing == True:

                if mode == True:

                    P_ROI.append([x, y])

                    # print(P_ROI)
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

                    # Get the pixel coordinates within the circle radius
                    r = 5
                    x_center = x
                    y_center = y

                    idx = np.where( np.sqrt(np.square(xv - x) + np.square(yv - y)) <= r )

                    idx_x = idx[0]
                    idx_y = idx[1]

                    pixel_circle = np.zeros((len(idx_x), 2))
                    pixel_circle[:, 0] = idx_x
                    pixel_circle[:, 1] = idx_y
                    # print(pixel_circle)

                    temp = P_circle
                    P_circle = np.append(temp, pixel_circle, axis = 0)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    img = cv2.imread('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Test_img.jpg', 1)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    [h, w] = img_grey.shape

    # Generate the pixel grid
    x = np.linspace(0, h-1, h)
    y = np.linspace(0, w-1, w)
    xv, yv = np.meshgrid(y, x)

    img_mask = np.zeros((img_grey.shape))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == 27:
            break

    # Mask (of course replace corners with yours)
    points = np.array([P_ROI], dtype = np.int32)
    points_circle = P_circle

    pixel_use = np.array(points_circle, dtype=np.int32)

    pixel_use = pixel_use[1:,:]

    points = pixel_use

    h = img.shape[0]
    w = img.shape[1]

    mask = np.zeros(img_grey.shape, dtype = np.uint8)

    for item in pixel_use:
        mask[item[0], item[1]] = 255

    # Apply the mask
    masked_image = cv2.bitwise_and(img_grey, mask)

    # Return the mask indices
    mask_idx = np.where(masked_image != 0)
    x_roi = mask_idx[1]
    y_roi = mask_idx[0]
    ROI = np.zeros([len(x_roi), 2])
    ROI[:,0] = x_roi
    ROI[:,1] = y_roi
    ROI = np.array(ROI, dtype = np.int32)

    pos_seed = np.mean(ROI, 0)
    # print(pos_seed)

    # cv2.imwrite('C:/Users/gm143/Documents/MATLAB/BTL/Data/exp_1/pos_12/before/mask.png', mask)
    cv2.imshow('masked image', masked_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return ROI, pos_seed

if __name__ == "__main__":

    # Read the image
    img_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Test_img.jpg"
    img = cv2.imread(img_path, 0)
    plt.imshow(img)
    plt.show()
    print("The image array is ", img)
    print(type(img))
    print(img.shape)

    # 3D Loop
    ROI, pos_seed = VS_draw(img)
    N = 300

    mask = np.zeros(img.shape, dtype = np.uint8)
    cv2.fillPoly(mask, np.int32([ROI]), 255)
    img_mask = cv2.bitwise_and(img, mask)
    plt.imshow(img_mask)
    plt.show()

    PC = []     # The object of 3D point cloud
    COLOR = []  # The color of the point cloud
    arr = np.array([150, 150, 1])   # Initialize

    # for i in range(1):
    img_obj = img                   # Capture the image
    for i in range(N):
        pc_obj = np.zeros((len(ROI), 3))
        x = ROI[:, 0]
        y = ROI[:, 1]
        z = np.ones((len(ROI))) * (i) # Get the z coordinate in the world coordinate
        pc_obj[:,0] = x
        pc_obj[:,1] = y
        pc_obj[:,2] = z
        print(pc_obj)
        print(pc_obj.shape)
        arr = np.vstack((arr, pc_obj))
        print(arr.shape)

    # Show the point cloud
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(arr)
    open3d.draw_geometries([pcd])

    # Show the slices
    # img_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Test_img.jpg"
    # img = cv2.imread(img_path, 0)
    # N = 200
    # IMG = []
    # for i in range(N):
    #     IMG.append(img)
    # img_stack = np.stack(IMG, axis=0)
    # print(len(IMG))
    # print(img_stack.shape)
    # sample_stack(img_stack)