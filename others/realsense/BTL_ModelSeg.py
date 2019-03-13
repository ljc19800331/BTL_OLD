# For model segmentation
# possible tutorial:
# 1. image processing: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing
# 2. im: https://github.com/quqixun/BrainPrep
# pipepine: 1. load the images 2. segment the images for each image 3. volume rendering from scratch
import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt
import vtk

class BtlModelSeg:

    def __init__(self):
        self.Filename = 1

    def Brain_Seg(self):
        # brain segementation
        a = 1

    def ReadMha(self):

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

def vtk_show(renderer, width = 400, height = 300):
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