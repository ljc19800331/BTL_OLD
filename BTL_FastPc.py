# This code is used to process the realtime point cloud
import vtk
import pyrealsense2 as rs
import cv2
import numpy as np
import DataConvert as DC
import BTL_VIZ

class fastpc:

    def __init__(self):

        a = 1

    def GetVerticesTexture(self):

        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipe.start(config)
        pc = rs.pointcloud()
        pro = profile.get_stream(rs.stream.color)                           # Fetch stream profile for depth stream
        intr = pro.as_video_stream_profile().get_intrinsics()               # Downcast to video_stream_profile and fetch intrinsics
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            frames = pipe.wait_for_frames()
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            color_image = np.asanyarray(color.get_data())
            pc.map_to(color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            points = pc.calculate(depth)
            vtx = np.asanyarray(points.get_vertices())
            vtx_npy = DC.Bracket2npy(vtx)
            tex = np.asanyarray(points.get_texture_coordinates())

            # tex to numpy tex
            tex_npy = np.zeros((len(tex), 2))
            for idx, val in enumerate(tex):
                # print val
                tex_npy[idx, 0] = val[0]
                tex_npy[idx, 1] = val[1]

            flag_save = raw_input("Do you want to save all the images and other data?")
            if flag_save == 'yes':
                cv2.imwrite('C:/Users/gm143/TumorCNC_brainlab/BTL/color_img.png', color_image)
                np.save('C:/Users/gm143/TumorCNC_brainlab/BTL/MTI_vertices.npy', vtx_npy)
                np.save('C:/Users/gm143/TumorCNC_brainlab/BTL/MTI_textures.npy', tex_npy)
                print "Save all the data"

            # print "The vertice coordinates are ", vtx_npy
            # print "The texture coordinates are ", tex_npy

        finally:
            pipe.stop()

        return vtx_npy, tex_npy

    def GetVerticeFromPixel(self, x_width = 380, y_height = 270):
        # input: pixel coordinate
        # output: 3D coordinate
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe_profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Map depth to color
        depth_pixel = [x_width, y_height]  # Random depth pixel     this is similar to my method
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
        color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
        pipeline.stop()

    def ShowPc(self):

        # Show point cloud in near realtime
        # Adjust the camera view location
        img = cv2.imread('C:/Users/gm143/TumorCNC_brainlab/BTL/realsense/mask.png')
        [h, w, d] = img.shape
        vtx_npy, tex_npy = self.GetVerticesTexture()
        idx_remove = np.int_(np.where((vtx_npy[:, 0] == 0) & (vtx_npy[:, 1] == 0) & (vtx_npy[:, 2] == 0)))

        tex_npy = np.delete(tex_npy,  idx_remove, 0)
        vtx_npy = np.delete(vtx_npy, idx_remove, 0)

        tex_npy[:,0] = np.int_(tex_npy[:,0] * w)
        tex_npy[:,1] = np.int_(tex_npy[:,1] * h)

        idx_remove = np.int_(np.where((tex_npy[:, 0] > (w-1)) | (tex_npy[:, 0] < 0) | (tex_npy[:, 1] > (h-1)) | (tex_npy[:, 1] < 0)))
        tex_npy = np.int_(np.delete(tex_npy,  idx_remove, 0))
        vtx_npy = np.delete(vtx_npy, idx_remove, 0)

        print "The shape of vertice is ", vtx_npy.shape
        print "The shape of the texture is ", tex_npy
        print "The first remove idx is ", idx_remove.shape

        colorvec_new = img[tex_npy[:, 1], tex_npy[:, 0], :]

        # Show the target region with the point values
        x_laser = 327
        y_laser = 200

        dis = np.sqrt(np.square(tex_npy[:,0] - x_laser) + np.square(tex_npy[:,1] - y_laser))
        idx_target = np.int_(np.where(dis < 2))
        print "The idx_target is ", idx_target
        tex_npy_target = tex_npy[idx_target, :]
        vtx_npy_target = np.reshape(vtx_npy[idx_target, :], (len(idx_target[0]), 3))
        colorvec_target = np.reshape(colorvec_new[idx_target, :], (len(idx_target[0]), 3))
        colorvec_target[:, 0] = 255
        colorvec_target[:, 1] = 0
        colorvec_target[:, 2] = 0

        # Apply transformation
        R_final = np.asarray([[0.99924, -0.0060221, -0.03842],
                              [0.0072509, -0.94176, 0.3362],
                              [-0.038208, -0.33622, -0.94101]])
        t_final = np.asarray([3.6437, 6.2995, 47.132])

        vtx_npy = vtx_npy * 100
        vtx_tform = np.matmul(vtx_npy, R_final) + np.tile(t_final, (len(vtx_npy), 1))
        vtx_npy_target = vtx_npy_target * 100
        vtx_npy_target_tform = np.matmul(vtx_npy_target, R_final) + np.tile(t_final, (len(vtx_npy_target), 1))

        # print "The vtx is ", vtx_npy
        print "The target_tform is ", vtx_npy_target_tform

        Actor_ROI_tform = BTL_VIZ.ActorNpyColor(vtx_tform, colorvec_new)
        Actor_target_tform = BTL_VIZ.ActorNpyColor(vtx_npy_target_tform, colorvec_target)
        # Actor_ROI = BTL_VIZ.ActorNpyColor(vtx_npy, colorvec_new)
        # # Actor_plane = BTL_VIZ.ActorNpyColor(Pc, Color_Vec)
        # BTL_VIZ.VizActor([Actor_ROI, Actor_target, Actor_target_tform])
        # vtk_Pc = npy2vtk(Pc_grid)
        # BTL_VIZ.VizVtk([vtk_Pc])

        print "The color vector is ", colorvec_new.shape

        return Actor_ROI_tform, Actor_target_tform

    def PcViz(self):

        # Visualization of the point cloud in a fast way
        Actor_ROI_tform, Actor_target_tform = self.ShowPc()

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        transform = vtk.vtkTransform()
        transform.Translate(1.0, 0.0, 0.0)

        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)

        transform = vtk.vtkTransform()
        # transform.Translate(0.0, 0.0, 0.0)
        transform.RotateX(180)
        transform.RotateY(180)
        transform.RotateZ(180)
        transform.Translate(0.0, 0.0, 0.0)

        # Design the first axes
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        renderer.AddActor(axes)

        # Design the camera object
        camera = vtk.vtkCamera()
        campos = np.zeros(3)
        campos[0] = 0
        campos[1] = 0
        campos[2] = 30
        camera.SetPosition(campos[0],  campos[1], campos[2] * 1.5)
        camera.SetFocalPoint(campos[0],  campos[1], campos[2])
        renderer.SetActiveCamera(camera)

        # renderer.AddActor(axes)
        renderer.AddActor(Actor_ROI_tform)
        renderer.AddActor(Actor_target_tform)
        # renderer.ResetCamera()
        renderWindow.Render()
        renderWindowInteractor.Start()

if __name__ == "__main__":

    test = fastpc()
    # test.GetVerticesTexture()
    # test.ShowPc()
    test.PcViz()