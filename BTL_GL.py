# This is used for textured mapping based colorized point cloud generation
'''
1. plane colorized point cloud -- finished
        Create a plane point cloud -- finished
        divide the region into several parts -- finished
2. non-plane colorized point cloud -- hemisphere model
3. more
4. Brain-based colorized point cloud
5. OpenGL for textured mapping -- possible solution

The current work model:
1. The openGL based textured model
2. The camera pose
3. Reconstruct a surface based on a sparse point cloud system

Code structure:
1. function -- textured image
2. function -- show the point cloud
3. function -- ImgMapPc
4. Define the camera pose model -- search for resource
5. The camera viewpoint pose prediction for the surgical site

The camera viewpoint
1. Capture the image from virtual camera
2. Divide the region -- within the image region
3. Image post processing -- extract the vessel and image -- to create the vessel network for registration
4. Transfer learning and synthetic data modelling

Reference:
1. Textured mapping in Python: https://plot.ly/~empet/14172/mapping-an-image-on-a-surface/#/
2. https://gist.github.com/somada141/acefac8a6360cba21f3a   -- the most important reference
3. KDtree - https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates

Problem:
1. resample the brain cortical surface -- add the number of points
2. The speed of taking the screen shots
3. Dense point cloud method
4. Preprocessing for the stl file -- this is im
5. Try with opengl -- this is im -- possibility of doing that
6. Blender for UV texture mapping
7. stl based method for stl textured mapping

Next step:
1. Change of the camera view point

'''
import cv2
import open3d
import vtk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from BTL_DataConvert import *
import BTL_VIZ
from BTL_VIZ import *
import BTL_GL
from scipy.spatial import cKDTree
import vtki

class BTL_GL():

    def __init__(self):
        self.img_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain_texture.jpg"
        self.pc_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/others/brain.ply"
        self.stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/hemis_150.stl'
        self.colorstl_file = '/home/mgs/PycharmProjects/BTL_GS/BTL/ColorRegimg_stl.npy'

        self.savepath_color = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/color_plane.npy'
        self.savepath_xyz = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/xyz_plane.npy'

    def GL_NonFlat(self):

        # Non-flat plane
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape
        img = np.fliplr(img[0: w - 1, 0: h - 1])
        img = np.fliplr(img[0: w - 1, 0: h - 1])
        print("The image shape is ", img.shape)
        pc = 1

        # Read the point cloud
        pc, xyz = self.TexPc()
        points = xyz[:, 0:2]
        values = xyz[:, 2]
        print("The shape of the point cloud is ", points.shape)
        print("The shape of values is ", values.shape)

        x_max = np.max(xyz[:, 0])
        y_max = np.max(xyz[:, 1])
        print("The maximum of x is ", x_max)
        print("The maximum of y is ", y_max)
        grid_x, grid_y = np.mgrid[0: x_max: complex(w - 1), 0: y_max: complex(h - 1)]

        # print("The grid_x is ", grid_x)
        # print("The grid y is ", grid_y)

        grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

        pc = np.zeros((len(grid_x.flatten()), 3))
        pc[:, 0] = grid_x.flatten()
        pc[:, 1] = grid_y.flatten()
        pc[:, 2] = grid_z.flatten()
        print(pc.shape)

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc)

        # Assign the color image
        [w, h, d] = img.shape
        # print("The width of the image is ", w)
        # print("The height of the image is ", h)

        vec_color = img.reshape(w * h, 3)
        pc_color = vec_color
        # vec_color = open3d.Vector3dVector(np.asarray(vec_color / 255))
        print("The shape of vec color is ", vec_color.shape)

        # Generate the mesh
        MeshGrid = [grid_x, grid_y, grid_z]

        # Generate the image matrix
        img_color = img

        # np.save('test_z.npy', grid_z)
        # print("The color image is ", img_color)
        # pcd.colors = vec_color
        # open3d.draw_geometries([pcd])

        return pcd, pc, pc_color, MeshGrid, img_color

    def GL_Flat(self):

        # Create the flat plane of the object
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape
        img = np.fliplr(img[0:w-1, 0:h-1])
        print(img)
        pc = 1
        x_max = 199
        y_max = 199
        grid_x, grid_y = np.mgrid[0:x_max:complex(w-1), 0:y_max:complex(h-1)]
        print("grid_x is ", grid_x)
        print("grid_y is ", grid_y)
        grid_z = np.ones((len(grid_x.flatten()),1))
        pc = np.zeros((len(grid_x.flatten()), 3))
        pc[:,0] = grid_x.flatten()
        pc[:,1] = grid_y.flatten()
        pc[:,2] = grid_z.flatten()
        print(pc)
        print(pc.shape)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc)
        # open3d.draw_geometries([pcd])

        # Assign the color image
        [w, h, d] = img.shape
        print("The width of the image is ", w)
        print("The height of the image is ", h)
        vec_color = img.reshape(w * h, 3)
        print(vec_color/255)
        # print(vec_color[:,0])
        vec_color = open3d.Vector3dVector(np.asarray(vec_color / 255))
        pcd.colors = vec_color
        open3d.draw_geometries([pcd])

    def GL_write(self, pcd_file, output_path):
        # Write the file
        pcd = pcd_file
        open3d.write_point_cloud(output_path, pcd)

    def GL_Cam(self):

        '''
        # Code for showing the colorized point cloud
        # The camera view point capture the image from different pose
        # Load the data and convert the data to vtk object
        # pcd, pc, pc_color = self.GL_NonFlat()
        # open3d.write_point_cloud("/home/mgs/PycharmProjects/BTL_GS/BTL_Data/TEST.ply", pcd)
        # vtk_obj = npy2vtk(pc)
        # Actor_pc = ActorNpyColorMesh(pc, pc_color)
        # Actor_pc = ActorNpyColor(pc, pc_color)
        # VizActor([Actor_pc])
        '''

        # Define the actor
        # Actor_pc = self.ActorColorPc()
        Actor_pc = self.ShowColorSTL()

        # Assign the pre-processing image -- if possible -- this is im
        N_pose = 5

        for i in range(N_pose):
            print(i)
            # Define the position of the camera and pose
            camera = vtk.vtkCamera()
            pos = 50 + 0.05 * i
            camera.SetPosition(pos, pos, pos - 0.05 * i)
            camera.SetFocalPoint(pos - 1, pos - 1, pos -1)
            # VizActorCam([Actor_pc], Cam = camera)
            # Get the screen shot
            VizActorCamShot([Actor_pc], Cam = camera , N_shot = i)

    def ActorColorPc(self):

        # Load the data object
        pcd, pc, pc_color, MeshGrid, img_color = self.GL_NonFlat()

        # Test the Color object
        # pc_color = np.load('/home/mgs/PycharmProjects/BTL_GS/BTL/Color_stl.npy')

        # Assign the points
        points = vtk.vtkPoints()
        for i in range(len(pc)):
            points.InsertNextPoint(pc[i, 0], pc[i, 1], pc[i, 2])
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Assign the colorized array to this case
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetNumberOfTuples(polydata.GetNumberOfPoints())
        for i in range(len(pc_color)):
            colors.InsertTuple3(i, pc_color[i, 0], pc_color[i, 1], pc_color[i, 2])

        # Connect the point object to the color object
        polydata.GetPointData().SetScalars(colors)

        # Sphere for mapping object
        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(5)
        sphere.SetThetaResolution(5)
        sphere.SetRadius(0.0025 * 10)
        sphere.Update()

        # Define the glyph object
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetColorModeToColorByScalar()
        glyph.SetVectorModeToUseNormal()
        glyph.ScalingOff()
        glyph.Update()

        # Define the mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        # Define the actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def TexImg(self):

        # Generate the textured image
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)    # Notice: BGR to RGB image
        plt.imshow(img)
        plt.show()

        # The binary classification -- this

        # Divide the region together -- this is im

        # The region -- this is im

    def TexPc(self):
        pc = open3d.read_point_cloud(self.pc_path)
        xyz = np.asarray(pc.points)
        print("The number of points is ", len(xyz))
        # open3d.draw_geometries([pc])
        return pc, xyz

    def TexMap(self):
        # Map the image to the region
        a = 1

    def AssignColor(self):

        '''
        # Assign color to the stl object and all the points
        '''

        # Load the data object
        pcd, pc, pc_color, MeshGrid, img_color = self.GL_NonFlat()

        # Load the stl

    def Color2STL(self):

        # Show the result for this project
        sr = vtk.vtkSTLReader()
        sr.SetFileName(self.stl_path)
        sr.Update()
        p = [0, 0, 0]
        p_list = []
        polydata = sr.GetOutput()

        # Extract the points from the object
        pc = np.zeros((polydata.GetNumberOfPoints(), 3))
        for i in range(polydata.GetNumberOfPoints()):
            print(i)
            polydata.GetPoint(i, p)
            tem = p[:]
            pc[i, 0] = tem[0]
            pc[i, 1] = tem[1]
            pc[i, 2] = tem[2]

        print("Test")
        # Get the color vector
        _,  _, pc_color, _, _ = self.GL_NonFlat()
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape

        # Dense grid points
        points = pc[:, 0:2]
        values = pc[:, 2]

        print("The shape of points is ", points.shape)

        x_min = np.min(pc[:, 0])
        x_max = np.max(pc[:, 0])
        y_min = np.min(pc[:, 1])
        y_max = np.max(pc[:, 1])

        z_max = np.max(pc[:, 2])

        grid_x, grid_y = np.mgrid[x_min: x_max: complex(w - 1), y_min: y_max: complex(h - 1)]

        print("The min value of x is ", x_min)
        print("The min value of y is ", y_min)
        print("The max value of x is ", x_max)
        print("The max value of y is ", y_max)

        print("The shape of the values is ", values.shape)
        print("The shape of the grid_x is ", grid_x.shape)
        print("The shape of the grid_y is ", grid_y.shape)

        print("The point is ", points)
        print("The value is ", values)

        grid_z = griddata(np.asarray(points), np.asarray(values), (grid_x, grid_y), method = 'nearest')

        print("Test")
        vec_color = pc_color
        xyz = np.zeros((len(grid_x.flatten()), 3))
        xyz[:, 0] = grid_x.flatten()
        xyz[:, 1] = grid_y.flatten()
        xyz[:, 2] = grid_z.flatten()

        # Show the two model
        pcd_stl = open3d.PointCloud()
        pcd_stl.points = open3d.Vector3dVector(pc)
        pcd_mesh = open3d.PointCloud()
        pcd_mesh.points = open3d.Vector3dVector(xyz)
        pcd_stl.paint_uniform_color([1, 0, 1])
        pcd_mesh.paint_uniform_color([1, 1, 0])
        open3d.draw_geometries([pcd_mesh, pcd_stl])

        # Define the STL object to obtain the 3D coordinates
        f = vtk.vtkSTLReader()
        f.SetFileName(self.stl_path)
        f.Update()  # This is necessary to have the data ready to read.
        obj = f.GetOutputDataObject(0)
        min_z, max_z = obj.GetBounds()[4:]
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(min_z, max_z)
        lut.Build()
        heights = vtk.vtkDoubleArray()
        heights.SetName("Z_Value")

        COLOR = np.zeros((len(pc), 3))

        # Assign the KDtree
        x_raven = xyz[:, 0]
        y_raven = xyz[:, 1]
        z_raven = xyz[:, 2]
        KdTree = cKDTree(np.c_[x_raven, y_raven, z_raven])
        print("Test")
        for i in range(len(pc)):
            print(i)
            p_obj = pc[i, :]
            dd, ii = KdTree.query([p_obj], k=1)
            idx_color = np.array(ii[0])
            vec_color = pc_color[idx_color, :]

            # Assign the color vector with the color index
            COLOR[i, :] = vec_color

        # Save the object
        np.save(self.savepath_color, COLOR)
        np.save(self.savepath_xyz, xyz)
        # return xyz

    def ShowColorSTL(self):

        # Show the colorized stl file
        # Generate the stl model -- we need to first know the color vector of the objects
        f = vtk.vtkSTLReader()
        f.SetFileName(self.stl_path)
        f.Update()

        # Update the frame rate
        obj = f.GetOutputDataObject(0)
        min_z, max_z = obj.GetBounds()[4:]

        lut = vtk.vtkLookupTable()
        lut.SetTableRange(min_z, max_z)
        lut.Build()

        heights = vtk.vtkDoubleArray()
        heights.SetName("Z_Value")

        # Load the color object
        Colors = vtk.vtkUnsignedCharArray()
        Colors.SetNumberOfComponents(3)
        Colors.SetName("Colors")

        # Load the color stl file (N x 3 vector)
        COLOR = np.load(self.colorstl_file)

        for i in range(obj.GetNumberOfPoints()):
            # z = obj.GetPoint(i)[-1]
            Colors.InsertNextTuple3(COLOR[i, 0], COLOR[i, 1], COLOR[i, 2])

        obj.GetPointData().SetScalars(Colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputDataObject(obj)
        mapper.SetScalarRange(min_z, max_z)
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(.1, .2, .4)

        renw = vtk.vtkRenderWindow()
        renw.AddRenderer(renderer)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renw)

        renw.Render()
        iren.Start()

        return actor, obj

    def RealtimeVideo(self):

        # Load the stl numpy data
        brain_color_npy = np.load('/home/mgs/PycharmProjects/BTL_GS/BTL/Colornpy_stl.npy')
        print("The brain color numpy is ", brain_color_npy)
        brain_gray_up = np.load('/home/mgs/PycharmProjects/BTL_GS/BTL/Graynpy_stl.npy')

        # Design the scene within the object
        # Define the transform system
        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        # Set the camera
        camera_1 = vtk.vtkCamera()
        camera_1.SetPosition(1, 1, 1)
        camera_1.SetFocalPoint(0, 0, 0)
        camera_2 = vtk.vtkCamera()
        camera_2.SetPosition(1, 1, 1)
        camera_2.SetFocalPoint(0, 0, 0)

        # Define the stl actor
        self.colorstl_file = '/home/mgs/PycharmProjects/BTL_GS/BTL/Colorimg_stl.npy'
        actor_color = self.ShowColorSTL()
        self.colorstl_file = '/home/mgs/PycharmProjects/BTL_GS/BTL/Grayimg_stl.npy'
        actor_grey = self.ShowColorSTL()

        # Design the renderer
        ren_1 = vtk.vtkRenderer()
        ren_1.SetActiveCamera(camera_1)
        renWin_1 = vtk.vtkRenderWindow()
        renWin_1.AddRenderer(ren_1)
        renWin_1.SetSize(500, 500)
        renWin_1.SetWindowName("Real in-vivo camera")

        ren_2 = vtk.vtkRenderer()
        ren_2.SetActiveCamera(camera_2)
        renWin_2 = vtk.vtkRenderWindow()
        renWin_2.AddRenderer(ren_2)
        renWin_2.SetSize(500, 500)
        renWin_2.SetWindowName("Virtual camera")

        ren_3 = vtk.vtkRenderer()
        # ren_3.SetActiveCamera(camera_2)
        renWin_3 = vtk.vtkRenderWindow()
        renWin_3.AddRenderer(ren_3)
        renWin_3.SetSize(500, 500)
        renWin_3.SetWindowName("The whole scene")

        # Add the actors and renders
        # ren_1.AddActor(axes)
        ren_1.AddActor(actor_color)
        # ren_2.AddActor(axes)
        ren_2.AddActor(actor_grey)

        ren_3.AddActor(actor_color)

        # The position of the camera relative to the focal point
        campos = np.zeros(3)
        delta = 0
        count = 0

        viz_tool = BTL_VIZ.VtkSimulateScan()
        xv, yv = viz_tool.ScanPath()
        print("The xv is ", xv)
        print("The yv is ", yv)

        while (True):

            count += 1
            print(count)
            delta += 0.05

            # Read the index in the scan path
            x_center = xv[count]
            y_center = yv[count]
            print("The x_center is ", x_center)
            print("The y center is ", y_center)
            vtk_spot, npy_spot = viz_tool.PointUpdate(brain_color_npy, x_center, y_center)
            print("The npy_spot is ", npy_spot)
            actor_spot = ActorNpyColor(npy_spot, vec_color=[255, 0, 0])

            # Convert the points to the visible spot -- a spherical ball model
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(np.mean(npy_spot[:, 0]), np.mean(npy_spot[:, 1]), np.mean(npy_spot[:, 2]))
            sphereSource.SetRadius(0.2)
            sphereSource.Update()
            mapper = vtk.vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor_ball = vtk.vtkActor()
            actor_ball.SetMapper(mapper)
            actor_ball.GetProperty().SetColor(1, 0, 0)  # (R,G,B)

            # Update the actor and camera
            campos[0] = np.mean(npy_spot[:, 0])
            campos[1] = np.mean(npy_spot[:, 1])
            campos[2] = np.mean(npy_spot[:, 2])

            print("The current camera position is ", campos)

            camera_1.SetPosition(campos[0], campos[1], campos[2] * 2)
            camera_1.SetFocalPoint(campos[0], campos[1], campos[2] * 1)
            camera_2.SetPosition(campos[0], campos[1], campos[2] * 2)
            camera_2.SetFocalPoint(campos[0], campos[1], campos[2] * 1)

            ren_1.SetActiveCamera(camera_1)
            ren_1.AddActor(actor_spot)
            ren_1.AddActor(actor_ball)
            renWin_1.Render()
            time.sleep(0.1)
            renWin_1.Render()

            ren_2.SetActiveCamera(camera_2)
            ren_2.AddActor(actor_spot)
            ren_2.AddActor(actor_ball)
            renWin_2.Render()
            time.sleep(0.1)
            renWin_2.Render()

            # ren_2.SetActiveCamera(camera_2)
            # ren_2.AddActor(actor_spot)
            # ren_2.AddActor(actor_ball)
            renWin_3.Render()
            time.sleep(0.1)
            renWin_3.Render()

    def ColorTextureMap(self, img_path, stl_path):

        '''
        1. solve from scratch
        2. obtain the points and color scalar and generate a new polydata object
        '''

        texture = vtki.load_texture(img_path)
        obj = vtki.read(stl_path)
        obj.texture_map_to_plane(inplace=True)
        obj.plot(texture = texture)

        # Mapper = vtk.vtkPolyDataMapper()
        # Mapper.SetInputData(obj)
        # Actor = vtk.vtkActor()
        # Actor.SetMapper(Mapper)
        #
        # renderer = vtk.vtkRenderer()
        # renderer.AddActor(Actor)
        # renWin = vtk.vtkRenderWindow()
        # renWin.AddRenderer(renderer)
        #
        # iren = vtk.vtkRenderWindowInteractor()
        # iren.SetRenderWindow(renWin)
        #
        # renWin.Render()
        # iren.Start()

        return obj

    def CamModel(self):

        # Define the camera model
        a = 1

if __name__ == "__main__":

    test = BTL_GL()
    # test.img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/lena.jpg'
    # test.pc_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/others/brain.ply'
    # test.stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_200.stl'
    # test.colorstl_file = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/color_plane.npy'
    # test.savepath_color = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/color_hemis.npy'
    # test.savepath_xyz = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/xyz_hemis.npy'

    # test.RealtimeVideo()
    # test.TexPc()
    # test.GL_Cam()
    # test.GL_HemisModel()
    # test.GL_NonFlat()
    # test.GL_Flat()
    # test.TexImg()
    # test.TexPc()
    # test.Color2STL()
    # test.ShowColorSTL()

    img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain_mask_color.jpg'
    stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/plane_stl_200.stl'
    test.ColorTextureMap(img_path, stl_path)