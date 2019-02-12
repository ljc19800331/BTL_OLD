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

Problem:
1. resample the brain cortical surface -- add the number of points
2. The speed of taking the screen shots
3. Dense point cloud method
4. Preprocessing for the stl file -- this is im
5. Try with opengl -- this is im -- possibility of doing that
6. Blender for UV texture mapping
7. stl based method for stl textured mapping
'''
import cv2
import open3d
import vtk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from BTL_DataConvert import *
from BTL_VIZ import *
import BTL_GL
from scipy.spatial import cKDTree

class BTL_GL():

    def __init__(self):
        self.img_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_Cortical_Scale.jpg"
        self.pc_path = "/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain.ply"
        self.stl_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/brain_slice.stl'

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
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)    # Notice: BGR to RGB image
        plt.imshow(img)
        plt.show()

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
            polydata.GetPoint(i, p)
            tem = p[:]
            pc[i, 0] = tem[0]
            pc[i, 1] = tem[1]
            pc[i, 2] = tem[2]

        _,  _, pc_color, _, _ = self.GL_NonFlat()
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        [w, h, d] = img.shape

        # Dense grid points
        points = pc[:, 0:2]
        values = pc[:, 2]

        x_min = np.min(pc[:, 0])
        x_max = np.max(pc[:, 0])
        y_min = np.min(pc[:, 1])
        y_max = np.max(pc[:, 1])

        z_max = np.max(pc[:, 2])

        print("The max value of x is ", x_max)
        print("The max value of y is ", y_max)

        grid_x, grid_y = np.mgrid[x_min: x_max: complex(w - 1), y_min: y_max: complex(h - 1)]
        grid_z = griddata(points, values, (grid_x, grid_y), method = 'nearest')

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

        for i in range(len(pc)):
            print(i)

            # Check the order of the points
            # Develop the height information for the model
            # if i < 40000:
            #     z = 0
            # else:
            #     z = obj.GetPoint(i)[-1]
            # heights.InsertNextValue(z)

            # The color mapping method
            # print(i)
            # p_obj = pc[i, :]
            # dis = np.absolute(xyz - p_obj)
            # dis_square = np.square(dis)
            # dis_sum = dis_square.sum(axis = 1)
            # dis_sqrt = np.sqrt(dis_sum)
            # idx = np.where(dis_sqrt == dis_sqrt.min())
            # idx_color = np.array(idx[0])
            # vec_color = pc_color[idx_color[0], :]
            # print(vec_color)

            p_obj = pc[i, :]
            dd, ii = KdTree.query([p_obj], k=1)
            # print(ii)
            idx_color = np.array(ii[0])
            # print(idx_color)
            vec_color = pc_color[idx_color, :]

            # Assign the color vector with the color index
            COLOR[i, :] = vec_color

        # Add this array to the point data as a scalar.
        # obj.GetPointData().SetScalars(heights)
        #
        # # Visualization stuff ... you need to tell the mapper about the scalar field
        # # and the lookup table. The rest of this is pretty standard stuff.
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputDataObject(obj)
        # mapper.SetScalarRange(min_z, max_z)
        # mapper.SetLookupTable(lut)
        #
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        #
        # renderer = vtk.vtkRenderer()
        # renderer.AddActor(actor)
        # renderer.SetBackground(.1, .2, .4)
        #
        # renw = vtk.vtkRenderWindow()
        # renw.AddRenderer(renderer)
        #
        # iren = vtk.vtkRenderWindowInteractor()
        # iren.SetRenderWindow(renw)
        #
        # renw.Render()
        # iren.Start()

        np.save('TestStl.npy', COLOR)

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

        COLOR = np.load('/home/mgs/PycharmProjects/BTL_GS/BTL/TestStl.npy')
        # print(COLOR.shape)

        for i in range(obj.GetNumberOfPoints()):
            z = obj.GetPoint(i)[-1]
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

        return actor

if __name__ == "__main__":

    test = BTL_GL()
    # test.TexPc()
    # test.GL_Cam()
    # test.GL_HemisModel()
    # test.GL_NonFlat()
    # test.GL_Flat()
    # test.TexImg()
    # test.TexPc()
    test.ShowColorSTL()
    # test.Color2STL()