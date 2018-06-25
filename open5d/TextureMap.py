# Fast texture mapping
# ref: https://www.programcreek.com/python/example/58539/vtk.vtkUnsignedCharArray
# ref: https://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/Point
# ref: https://www.vtk.org/Wiki/VTK/Examples/Python/TriangleColoredPoints
import csv
import numpy as np
import cv2
import math
import SEC_VIZ
from SEC_VIZ import *
import matplotlib.pyplot as plt
import BTL_VIZ
from DataConvert import *

class Feature3d:

    def __init__(self):
        self.color_name = '/home/maguangshen/PycharmProjects/BTL_GS/open5d/data/color.png'
        self.tex_name = '/home/maguangshen/PycharmProjects/BTL_GS/open5d/data/tex.csv'
        self.vtx_name = '/home/maguangshen/PycharmProjects/BTL_GS/open5d/data/vtx.csv'

    def DataLoad(self):

        # Read the tex file
        f = open(self.tex_name, 'rb')
        reader = csv.reader(f, delimiter=',', quotechar='"')
        TEX = np.zeros((307200, 2))
        for idx, row in enumerate(reader):
            TEX[idx] = row

        # Read the tex file
        f = open(self.vtx_name, 'rb')
        reader = csv.reader(f, delimiter=',', quotechar='"')
        VTX = np.zeros((307200, 3))
        for idx, row in enumerate(reader):
            VTX[idx] = row

        # Read the color file
        COLOR = cv2.imread(self.color_name)

        return TEX, VTX, COLOR

    def Img_filter(self):

        # Segment the target region of the image
        x1 = 100; x2 = 450
        y1 = 150; y2 = 400
        img_color = cv2.imread(self.color_name)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_crop = img_color[y1 : y2, x1 : x2]
        # The shape of the image
        [h, w, d] = img_color.shape
        fig = plt.figure()
        fig_col = 2; fig_row = 2
        fig.add_subplot(fig_row, fig_col, 1)
        plt.imshow(img_crop)
        # index between image and point cloud
        # The ngrid of the image matrix
        yy, xx = np.mgrid[0 : h, 0 : w]
        # Return the index of the cropped region
        [idx_row, idx_col] = np.where((xx >= x1) & ( xx <= x2) & ( yy >= y1) & (yy <= y2))
        idx = sub2ind(idx_row, idx_col, h, w)

        # Return the index to the threshold
        img_region = np.zeros((h, w), np.uint8)       # notice: you have to set the type of the number
        for i in range(len(idx)):
            img_region[idx_row[i]][idx_col[i]] = img_gray[idx_row[i]][idx_col[i]]
        img_region.astype(int)
        fig.add_subplot(fig_row, fig_col, 2)
        plt.imshow(img_region)

        # Filter an image with the binary threshold which is larger then some thresholds
        (thresh, img_bw) = cv2.threshold(img_region, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # convert the other region to '1'
        [idx_row_1, idx_col_1] = np.asanyarray(np.where( (img_bw == 0) & (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)))
        idx_1 = sub2ind(idx_row_1, idx_col_1, h, w)     # This is the index for the rectangle region

        # Find the index of the object region, different from the target region
        # Return the index to the threshold
        img_obj = np.zeros((h, w), np.uint8)  # notice: you have to set the type of the number
        for i in range(len(idx_1)):
            img_obj[idx_row_1[i]][idx_col_1[i]] = 255
        img_obj.astype(int)
        fig.add_subplot(fig_row, fig_col, 3)
        plt.imshow(img_obj)
        plt.show()

        idx_region = idx  # The cropped region
        idx_target = idx_1  # The target region

        return idx_region, idx_target

    def ColorPc(self, idx_region, idx_target):

        # Load the texture and vertice point cloud
        test = Feature3d()
        TEX, VTX, COLOR = test.DataLoad()       # COLOR is a grayscale image
        color = cv2.imread(self.color_name)
        img_color = cv2.imread(self.color_name)
        [h, w, d] = img_color.shape

        # Self define the index of the coordinates
        idx = idx_target
        # idx = idx_region

        # Extract the target coordinates
        tex_u = TEX[:, 0]
        tex_v = TEX[:, 1]

        # The coordinates in the image
        W = np.floor(tex_u[idx] * w - 0.5)
        H = np.floor(tex_v[idx] * h - 0.5)

        # Vertex coordinates
        vtx = VTX[idx, :]

        # Color vector
        R = color[:, :, 0]; R_vec = np.reshape(R, ( h * w) )
        G = color[:, :, 1]; G_vec = np.reshape(G, ( h * w) )
        B = color[:, :, 2]; B_vec = np.reshape(B, ( h * w) )
        r = R_vec[idx]
        g = G_vec[idx]
        b = B_vec[idx]

        # Threshold the point cloud when its value is not in the range (0,1)
        W[ (W < 0) | (W == 0) | (W > w) ] = 1
        H[ (H < 0) | (H == 0) | (H > h) ] = 1

        # Show the point cloud with the vtk
        vtk_obj = BTL_VIZ.VtkPointCloud()
        for k in range(len(vtx)):
            point = (vtx[k][0], vtx[k][1], vtx[k][2])
            vtk_obj.addPoint(point)
        BTL_VIZ.VizVtk([vtk_obj])

        # The color vector
        # Input: (r,g,b); Output: the color vector on a 3D point
        # use ply to write the color vector into the ply file
        points = vtx
        color = np.zeros((len(r), 3))
        color[:, 0] = r[:]
        color[:, 0] = g[:]
        color[:, 0] = b[:]

        # Save the data to the ply file
        write_points_in_vtp(points, color, outfile='points.vtp')

        filename = '/home/maguangshen/PycharmProjects/BTL_GS/open5d/points.vtp'
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4)  # set background color
        renderer.ResetCamera()
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(reader.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        # Begin Interaction
        renderWindow.Render()
        renderWindowInteractor.Start()

def write_points_in_vtp(points, color, outfile='points.vtp'):

    # Set up points and vertices
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()

    # Set up the color vector
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")

    # Assign the values to the point object
    for i in range(points.shape[0]):
        ind = Points.InsertNextPoint(points[i][0], points[i][1], points[i][2])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(ind)
        Colors.InsertNextTuple3(color[i][0], color[i][1], color[i][2])

    # Define the poly data object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    polydata.SetVerts(Vertices)
    polydata.GetPointData().SetScalars(Colors)
    polydata.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

if __name__ == '__main__':

    # Generate the colorized point cloud
    # Class object
    test = Feature3d()

    # Load the data
    TEX, VTX, COLOR = test.DataLoad()

    # Load the color filter
    idx_region, idx_target = test.Img_filter()

    # generate the color point cloud
    test.ColorPc(idx_region, idx_target)