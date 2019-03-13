# The helper function for Texture Mapping in Python
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d

# Define the grid
x = np.linspace(0, 5, 200)
y = np.linspace(5, 10, 200)
X, Y = np.meshgrid(x,y)
z = (X+Y)/(2+np.cos(X)*np.sin(Y))

img = plt.imread('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_cortical.jpg')
# plt.imshow(img, cmap = cm.Greys_r)
# plt.show()
# print('image shape', img.shape)
# surfcolor = np.fliplr(img[200:400, 200:400])

# def mpl_to_plotly(cmap, pl_entries):
#
#     # pl_entries : color of the scale level for the color map
#     h=1.0/(pl_entries-1)
#     pl_colorscale=[]
#     for k in range(pl_entries):
#         C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
#         pl_colorscale.append([round(k*h,2), 'rgb'+str((C[0], C[1], C[2]))])
#     return pl_colorscale

# pl_grey = mpl_to_plotly(cm.Greys_r, 21)

surf = Surface(x=x, y=y, z=z,
             colorscale=pl_grey,
             surfacecolor=surfcolor,
             showscale=False)

x_max = 5
y_max = 5
grid_x, grid_y = np.mgrid[0:x_max:200j, 0:y_max:200j]
grid_z = np.ones((len(grid_x.flatten()),1))
pc = np.zeros((len(grid_x.flatten()), 3))

pc[:,0] = grid_x.flatten()
pc[:,1] = grid_y.flatten()
pc[:,2] = grid_z.flatten()

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(pc)

vec_color = surf.surfacecolor.reshape(200 * 200, 3)
vec_color = open3d.Vector3dVector(vec_color)
pcd.colors = vec_color
open3d.draw_geometries([pcd])

print(surf.x)
print(surf.surfacecolor.shape)
# print(vec_color.shape)

# Initialization
# noaxis = dict(showbackground=False,
#               showgrid=False,
#               showline=False,
#               showticklabels=False,
#               ticks='',
#               title='',
#               zeroline=False)
#
# layout = Layout(
#          title='Mapping an image onto a surface',
#          font=plotly.graph_objs.layout.Font(family='Balto'),
#          width=800,
#          height=800,
#          scene = Scene(xaxis=plotly.graph_objs.layout.scene.XAxis(noaxis),
#                      yaxis=plotly.graph_objs.layout.scene.YAxis(noaxis),
#                      zaxis=plotly.graph_objs.layout.scene.ZAxis(noaxis),
#                      aspectratio=dict(x=1,
#                                       y=1,
#                                       z=0.5
#                                      ),
#                     )
#         )
#
# fig = Figure(data=[surf], layout=layout)
# py.sign_in('empet', 'api_key')
# py.iplot(fig, filename = 'mappingLenaSurf')