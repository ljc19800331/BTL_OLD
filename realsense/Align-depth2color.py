import sys
import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# Different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print "Depth Scale is: ", depth_scale

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.3  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
print(clipping_distance)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Define the pointCloud object
pc = rs.pointcloud()

# Streaming loop
try:
    while True:

        # Get frame set of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        color_frame_test = frames.get_color_frame()

        pc.map_to(color_frame)
        points = pc.calculate(aligned_depth_frame)

        # print(vtx.shape)
        # print(tex.shape)

        # Validate that both frames are valid
        # if not aligned_depth_frame or not color_frame:
        #     continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())     # This is just 1D information
        color_image = np.asanyarray(color_frame.get_data())
        color_image_test = np.asanyarray(color_frame_test.get_data())

        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())

        # print(vtx.shape)
        # print(tex.shape)
        # np.savetxt("vtx.csv", vtx, delimiter=",")
        # np.savetxt("tex.csv", tex, delimiter=",")
        # cv2.imwrite('color.png', color_image)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153

        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))     # depth image is 1 channel, color is 3 channels
        # print(depth_image_3d)
        # print(clipping_distance)
        bg_removed = np.where( (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),  grey_color,   color_image)        # This is just an image not a point cloud
        # print((depth_image_3d > clipping_distance) | (depth_image_3d <= 0))
        # print(bg_removed.shape)      # [153, 153, 153] for each row
        # cv2.imwrite('color.png', color_image)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # print(bg_removed)
        images = np.hstack((color_image, color_image_test))
        # images = np.hstack((bg_removed, depth_colormap))                # bg_removed: (480,640,3) for image
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        cv2.waitKey(1)

finally:
    pipeline.stop()