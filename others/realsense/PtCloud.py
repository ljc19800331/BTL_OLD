# First import the library
import sys
sys.path.append('/usr/local/lib')
import numpy as np
import pyrealsense2 as rs
import scipy.io
import cv2
import DataConvert
from DataConvert import *

# Create a pipeline
pipe = rs.pipeline()

# Create a config and configure the pipeline to stream
# Different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipe.start(config)
pc = rs.pointcloud()

pro = profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
intr = pro.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
# print intr.ppx

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

try:

# while (True):
    # Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames()

    # Fetch color and depth frames
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    color_image = np.asanyarray(color.get_data())

    # Tell pointcloud object to map to this color frame
    pc.map_to(color)
    # print(color)

    # Depth align image -- different from the color image
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # color_alignframe = aligned_frames.get_depth_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # color_alignimage = np.asanyarray(color_alignframe.get_data())
    # color_alignimage_3d = np.dstack((color_alignimage, color_alignimage, color_alignimage))

    # Design the mask image
    cv2.imshow("images", color_image)
    cv2.waitKey(5)

    # Generate the pointcloud and texture mappings -- points is an object ...
    points = pc.calculate(depth)                    # This is just a point cloud without color overlaid
    # points_test = np.reshape(points)
    # print(points.shape)
    # print(type(points))

    # Export the texture and vertice coordinates
    vtx = np.asanyarray(points.get_vertices())
    npy_data = Bracket2npy(vtx)
    points_test = vtx
    print(npy_data.shape)
    print(npy_data)
    print(vtx)
    # print(type(vtx))
    # print(vtx.shape)
    # print(vtx.shape)
    # print(points_test)
    # print(vtx[1])
    # print(vtx)
    # print(points_test[0])
    # tex = np.asanyarray(points.get_texture_coordinates())
    # np.savetxt("vtx.csv", vtx, delimiter=",")
    # np.savetxt("tex.csv", tex, delimiter=",")
    # cv2.imwrite('color.png', color_image)
    # cv2.imwrite('color_align.png', depth_colormap)
    # print(color_alignimage_3d)
    # print(color_alignimage)

    # Export the colored point cloud
    # points.export_to_ply("PtCloud.ply", color)

finally:
    pipe.stop()

# Get the vertices data
# vtx = np.asanyarray(points.get_vertices())
# tex = np.asanyarray(points.get_texture_coordinates())
# print(type(vtx))
# print(tex.shape)
# scipy.io.savemat('vtx.mat', mdict={'vtx': vtx})
# scipy.io.savemat('tex.mat', mdict={'tex': tex})

# np.save('vtx.npy', vtx)
# np.save('tex.npy', tex)
# cv2.imwrite('color.png', color_image)

# print("Saving to RealsensePointCloud.ply...")
# points.export_to_ply("PtCloud.ply", color)          # Is this the color point cloud ?
# print("Done")