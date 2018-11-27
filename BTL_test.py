'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2
import glob
import vtk
from vtk import *
from vtk.util import numpy_support
from BTL_DataConvert import *
import pydicom
import os
import numpy
from matplotlib import pyplot, cm
import plotly
import plotly.graph_objs
from plotly.graph_objs import *
from IPython.display import Image
import SimpleITK
import csv
import BTL_VIZ
from BTL_VIZ import *
import open3d

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

class Test():

    def __init__(self):
        self.w = 640
        self.h = 480
    def write_ply(self, fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    def test_1(self):

        print('loading images...')
        imgL = cv2.pyrDown(cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/BTL/aloeL.jpg') )  # downscale images for faster processing
        imgR = cv2.pyrDown(cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/BTL/aloeR.jpg') )

        # disparity range is tuned for 'aloe' image pair
        window_size = 3
        min_disp = 16
        num_disp = 112 - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        print('generating 3d point cloud...',)
        h, w = imgL.shape[:2]
        f = 0.8 * w                           # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0, 1, 0, -0.5*h],  # turn points 180 deg around x-axis,
                        [0, 0, 0,     f],  # so that y-axis looks up
                        [0, 0, 1,     0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
        self.write_ply('out.ply', out_points, out_colors)
    def test_2(self):

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9 * 6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('/home/maguangshen/PycharmProjects/BTL_GS/BTL/Left/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        img = cv2.imread('/home/maguangshen/PycharmProjects/BTL_GS/BTL/Left/12.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('calibresult.png', dst)

        mean_error = 0
        tot_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        print("The length of the obj points", len(objpoints))
        print("total error: ", tot_error)
        print("mean error: ", tot_error/ len(objpoints))
    def test_3(self):

        filename = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainMra/stl_test.stl"

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)

        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        # Create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Assign actor to the renderer
        ren.AddActor(actor)

        # Enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()
    def test_4(self):
        name_vtk = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainMra/test.vtk'
        name_stl = 'stl_test.stl'
        vtk2stl(name_vtk, name_stl)
    def test_5(self):

        # The source file
        file_name = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainMra/test.vtk"

        # Read the source file.
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()
        scalar_range = output.GetScalarRange()

        # Create the mapper that corresponds the objects of the vtk.vtk file
        # into graphics elements
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(output)
        mapper.SetScalarRange(scalar_range)

        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)  # Set background to white

        # Create the RendererWindow
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.AddRenderer(renderer)

        # Create the RendererWindowInteractor and display the vtk_file
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renderer_window)
        interactor.Initialize()
        interactor.Start()
    def test_6(self):

        PathDicom = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/dicom'
        lstFilesDCM = []  # create an empty list
        # Save the file names for the object
        for dirName, subdirList, fileList in os.walk(PathDicom):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName, filename))
        # print(lstFilesDCM)

        # Get ref file -- All the information in the Dicom files
        RefDs = pydicom.read_file(lstFilesDCM[0])
        # print(RefDs)

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
        print(ConstPixelDims)

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        print(ConstPixelSpacing)
        x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
        z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])
        print(x.shape)
        print(y.shape)
        print(z.shape)

        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # Loop through all the dicoms to get the images
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

        pyplot.figure(dpi=300)
        pyplot.axes().set_aspect('equal', 'datalim')
        pyplot.set_cmap(pyplot.gray())
        pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 80]))
        pyplot.show()

        pyplot.figure(dpi=300)
        pyplot.axes().set_aspect('equal', 'datalim')
        pyplot.set_cmap(pyplot.gray())
        pyplot.pcolormesh(z, x, numpy.flipud(ArrayDicom[:, 100, :]))
        pyplot.show()
    def test_7(self):

        PathDicom = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/vhm_head/"
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(PathDicom)
        reader.Update()

        # Load dimensions using `GetDataExtent`
        _extent = reader.GetDataExtent()
        ConstPixelDims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1, _extent[5] - _extent[4] + 1]

        # Load spacing values
        ConstPixelSpacing = reader.GetPixelSpacing()

        # ArrayDicom -- read the CT data
        ArrayDicom = vtkImageToNumPy(reader.GetOutput(), ConstPixelDims)
        # plotHeatmap(numpy.rot90(ArrayDicom[:, 256, :]), name="CT_Original")

        data = numpy.rot90(ArrayDicom[:, 256, :])
        print(data)
        print(data.shape)
        pyplot.imshow(data)
        pyplot.show()

        # Threshold holder
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(reader.GetOutputPort())
        threshold.ThresholdByLower(400)  # remove all soft tissue
        threshold.ReplaceInOn()
        threshold.SetInValue(0)  # set all values below 400 to 0
        threshold.ReplaceOutOn()
        threshold.SetOutValue(1)  # set all values above 400 to 1
        threshold.Update()

        ArrayDicom = vtkImageToNumPy(threshold.GetOutput(), ConstPixelDims)
        data = numpy.rot90(ArrayDicom[:, 256, :])

        # Discrete Volume rendering process
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(threshold.GetOutputPort())
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(dmc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1.0, 1.0, 1.0)

        camera = renderer.MakeCamera()
        camera.SetPosition(-500.0, 245.5, 122.0)
        camera.SetFocalPoint(301.0, 245.5, 122.0)
        camera.SetViewAngle(30.0)
        camera.SetRoll(-90.0)
        renderer.SetActiveCamera(camera)
        vtk_show(renderer, 600, 600)

        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.AddActor(actor)
        renderer.SetBackground(.1, .2, .3)  # Background color dark blue
        renderer.SetBackground(.3, .2, .1)  # Background color dark red
        renderWindow.Render()
        renderWindowInteractor.Start()

        # camera = renderer.GetActiveCamera()
        # camera.SetPosition(301.0, 1045.0, 122.0)
        # camera.SetFocalPoint(301.0, 245.5, 122.0)
        # camera.SetViewAngle(30.0)
        # camera.SetRoll(0.0)
        # renderer.SetActiveCamera(camera)
        # vtk_show(renderer, 600, 600)
    def test_8(self):

        coneSource = vtk.vtkConeSource()
        # coneSource.SetResolution(60)
        # coneSource.SetCenter(-2,0,0)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(coneSource.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Visualize
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.AddActor(actor)
        renderer.SetBackground(.1, .2, .3)  # Background color dark blue
        renderer.SetBackground(.3, .2, .1)  # Background color dark red
        renderWindow.Render()
        renderWindowInteractor.Start()
    def test_9(self):

        pathDicom = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/MyHead/"
        idxSlice = 50
        labelWhiteMatter = 1
        labelGrayMatter = 2

        # Read the Dicom from the folder name
        reader = SimpleITK.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
        reader.SetFileNames(filenamesDICOM)
        imgOriginal = reader.Execute()

        # Viz one slice of the Dicom image
        imgOriginal = imgOriginal[:, :, idxSlice]
        sitk_show(imgOriginal)

        # CurvatureFlow filter
        imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                            timeStep=0.125,
                                            numberOfIterations=5)
        sitk_show(imgSmooth)

        # Initial segmentation of white matter -- threshold based method
        lstSeeds = [(150, 75)]
        imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,
                                                      seedList=lstSeeds,
                                                      lower=130,
                                                      upper=190,
                                                      replaceValue=labelWhiteMatter)

        # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
        print(imgWhiteMatter.GetPixelID)
        imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
        sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatter))

        # Fill in the holes with voting binary hole filter
        imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                                  radius=[2] * 3,
                                                                  majorityThreshold=1,
                                                                  backgroundValue=0,
                                                                  foregroundValue=labelWhiteMatter)
        sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))
    def test_10(self):

        # Read the mhd files
        # mhd file is only the title
        filenameT1 = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/mr_T1/patient_109_mr_T1.mhd"
        filenameT2 = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/mr_T2/patient_109_mr_T2.mhd"

        # Slice index to visualize with 'sitk_show'
        idxSlice = 26

        # int label to assign to the segmented gray matter
        labelGrayMatter = 1

        # Read the original data
        imgT1Original = SimpleITK.ReadImage(filenameT1)
        imgT2Original = SimpleITK.ReadImage(filenameT2)
        # print(imgT1Original)

        sitk_show(SimpleITK.Tile(imgT1Original[:, :, idxSlice],
                                 imgT2Original[:, :, idxSlice],
                                 (2, 1, 0)))
        # Smooth filter operation
        imgT1Smooth = SimpleITK.CurvatureFlow(image1=imgT1Original,
                                              timeStep=0.125,
                                              numberOfIterations=5)

        imgT2Smooth = SimpleITK.CurvatureFlow(image1=imgT2Original,
                                              timeStep=0.125,
                                              numberOfIterations=5)
        sitk_show(SimpleITK.Tile(imgT1Smooth[:, :, idxSlice],
                                 imgT2Smooth[:, :, idxSlice],
                                 (2, 1, 0)))

        lstSeeds = [(165, 178, idxSlice),
                    (98, 165, idxSlice),
                    (205, 125, idxSlice),
                    (173, 205, idxSlice)]

        imgSeeds = SimpleITK.Image(imgT2Smooth)
        for s in lstSeeds:
            imgSeeds[s] = 10000
        sitk_show(imgSeeds[:, :, idxSlice])
        print(imgSeeds)

        # Uni-modal Segmentation
        # Region growing on each of the two images based on the predefined seeds
        imgGrayMatterT1 = SimpleITK.ConfidenceConnected(image1=imgT1Smooth,
                                                        seedList=lstSeeds,
                                                        numberOfIterations=7,
                                                        multiplier=1.0,
                                                        replaceValue=labelGrayMatter)

        imgGrayMatterT2 = SimpleITK.ConfidenceConnected(image1=imgT2Smooth,
                                                        seedList=lstSeeds,
                                                        numberOfIterations=7,
                                                        multiplier=1.5,
                                                        replaceValue=labelGrayMatter)
        imgT1SmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgT1Smooth),
                                        imgGrayMatterT1.GetPixelID())
        imgT2SmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgT2Smooth),
                                        imgGrayMatterT2.GetPixelID())
        sitk_tile_vec([SimpleITK.LabelOverlay(imgT1SmoothInt[:, :, idxSlice],
                                              imgGrayMatterT1[:, :, idxSlice]),
                       SimpleITK.LabelOverlay(imgT2SmoothInt[:, :, idxSlice],
                                              imgGrayMatterT2[:, :, idxSlice])])

        # Multi-modal image segmentation
        imgComp = SimpleITK.Compose(imgT1Smooth, imgT2Smooth)
        imgGrayMatterComp = SimpleITK.VectorConfidenceConnected(image1=imgComp,
                                                               seedList=lstSeeds,
                                                               numberOfIterations=1,
                                                               multiplier=0.1,
                                                               replaceValue=labelGrayMatter)
        sitk_show(SimpleITK.LabelOverlay(imgT2SmoothInt[:, :, idxSlice],
                                         imgGrayMatterComp[:, :, idxSlice]))
    def test_11(self):
        # Show the images
        l2n = lambda l: numpy.array(l)
        n2l = lambda n: list(n)

        # Path to the .mha file
        filenameSegmentation = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/nac_brain_atlas/brain_segmentation.mha"

        # Path to colorfile.txt
        filenameColorfile = "/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/nac_brain_atlas/colorfile.txt"

        # Opacity of the different volumes (between 0.0 and 1.0)
        volOpacityDef = 0.25

        # Read the mha data
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(filenameSegmentation)
        castFilter = vtk.vtkImageCast()
        castFilter.SetInputConnection(reader.GetOutputPort())
        castFilter.SetOutputScalarTypeToUnsignedShort()
        castFilter.Update()
        imdataBrainSeg = castFilter.GetOutput()

        # Read the color file
        fid = open(filenameColorfile, "r")
        reader = csv.reader(fid)
        dictRGB = {}
        for line in reader:
            dictRGB[int(line[0])] = [float(line[2]) / 255.0,
                                     float(line[3]) / 255.0,
                                     float(line[4]) / 255.0]
    def test_12(self):
        # Viz the brain vessel and the vascular
        vesselname = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainData/brain_vessel.stl'
        vesselactor = ActorStl(vesselname)
        brainname = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainData/Full_brain.stl'
        brainactor = ActorStl(brainname)
        skullfillingname = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainData/Stroke Vessels.stl'
        skullfillactor = ActorStl(skullfillingname)
        VizActor([brainactor])

    def test_13(self):
    # Test the file change problem
        test = 1

    def test_14(self):
        pcd = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/Brain_color.ply")
        open3d.draw_geometries([pcd])

def createDummyRenderer():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    camera = renderer.MakeCamera()
    camera.SetPosition(-256, -256, 512)
    camera.SetFocalPoint(0.0, 0.0, 255.0)
    camera.SetViewAngle(30.0)
    camera.SetViewUp(0.46, -0.80, -0.38)
    renderer.SetActiveCamera(camera)

    return renderer

def sitk_tile_vec(lstImgs):
    lstImgToCompose = []
    for idxComp in range(lstImgs[0].GetNumberOfComponentsPerPixel()):
        lstImgToTile = []
        for img in lstImgs:
            lstImgToTile.append(SimpleITK.VectorIndexSelectionCast(img, idxComp))
        lstImgToCompose.append(SimpleITK.Tile(lstImgToTile, (len(lstImgs), 1, 0)))
    sitk_show(SimpleITK.Compose(lstImgToCompose))

def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

def plotHeatmap(array, name="plot"):

    data = ([Heatmap(z=array,scl='Greys')])
    print(data)
    print(data.shape)
    # data = plotly.graph_objs.Data([Heatmap(z=array,scl='Greys')])
    layout = Layout(autosize=False,title=name)
    fig = Figure(data=data, layout=layout)

    return plotly.plotly.iplot(fig, filename=name)

def vtkImageToNumPy(image, pixelDims):
    pointData = image.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')
    return ArrayDicom

def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    # writer = vtk.vtkPNGWriter()
    # writer.SetWriteToMemory(1)
    # writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    # writer.Write()
    # data = str(buffer(writer.GetResult()))

    # return Image(data)

if __name__ == "__main__":

    test = Test()
    test.test_14()