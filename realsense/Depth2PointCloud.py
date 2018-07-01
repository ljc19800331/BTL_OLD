# https://github.com/IntelRealSense/librealsense/issues/1904
# https://github.com/IntelRealSense/librealsense/issues/1274
# https://software.intel.com/sites/products/realsense/person/developer_guide.html
# https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#point-coordinates
import sys
sys.path.append('/usr/local/lib')
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import BTL_VIZ
from BTL_VIZ import *
import TextureMap
from TextureMap import *

def CaptureColorPc():

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipe_profile = pipeline.start(config)
    curr_frame = 0

    try:
        while(curr_frame < 20):

            # Get the color frame
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            # Get aligned frames for the color frame which is different from the color frame
            align_to = rs.stream.color
            align = rs.align(align_to)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()          # Align_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Wait for a coherent pair of frames: depth and color
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue

            # Intrinsic and extrinsic -- the imager of the color and depth frame is different -- this is important
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Visualize the realtime color image
            # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Align Example', color_image)
            # cv2.waitKey(10)
    #
            # This is important since the first frame of the color image is dark while later it is better
            curr_frame += 1
            if(curr_frame == 10):
                cv2.imwrite('/home/maguangshen/PycharmProjects/BTL/realsense/mask.png', color_image)

    finally:
        pipeline.stop()

    # Get the mask image where the laser spot region is covered with the black color and other with the white color
    img= cv2.imread('/home/maguangshen/PycharmProjects/BTL/realsense/mask.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    delta = 5
    w = delta
    h = delta
    offset_x = delta
    offset_y = delta
    x1 = maxLoc[0] - offset_x
    y1 = maxLoc[1] - offset_y
    height, width, depth = img.shape
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.rectangle(mask, (x1, y1), (x1 + 2 * w, y1 + 2 * h), (255, 255, 255), -1)
    masked_data = cv2.bitwise_and(img, mask)
    maskedImg = cv2.bitwise_and(color_image, masked_data)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    graymask = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)

    # viz the mask and gray image
    # cv2.imshow("images", np.hstack([grayimg, graymask]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Determine the region of the target region
    delta_region = 50
    x1_cols = x1 - delta_region
    x2_cols = x1 + delta_region
    y1_rows = y1 - delta_region
    y2_rows = y1 + delta_region

    # init the cols and rows
    cols = list(range(x1_cols, x2_cols))
    rows = list(range(y1_rows, y2_rows))

    # Generate the 3D point cloud and save to the np format
    Pc = np.zeros((len(cols) * len(rows), 3 ))
    Color_Vec = np.zeros((len(cols) * len(rows), 3))

    # Assign the point cloud to the image
    color_idx = []

    count = 0
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
            colorvec = maskedImg[np.int(i_idx), np.int(j_idx), :]

            # if the light is not in 0,0,0
            if(colorvec[0] != 0 and colorvec[1] != 0 and colorvec[2] != 0):
                color_idx.append(count)
                Pc[count, :] = depth_point
                Color_Vec[count, :] = colorvec

            count += 1

    # Viz the colorized point cloud
    Show_color_pc(Pc, Color_Vec)
    vtk_Pc = npy2vtk(Pc)
    VizVtk([vtk_Pc])

if __name__ == "__main__":

    CaptureColorPc()