'''
ref1: https://gist.github.com/Hodapp87/8874941
ref2: https://github.com/Kitware/VTK/blob/master/Examples/Modelling/Python/DelMesh.py

Solution to create a good surface:
1. vtkSurfaceReconstructionFilter -- https://github.com/Kitware/VTK/blob/master/Examples/Modelling/Python/reconstructSurface.py
2. https://gist.github.com/Jerdak/7364746 -- PyOpenGL textured mapping
3. https://pythonprogramming.net/opengl-pyopengl-python-pygame-tutorial/ -- PyOpenGL application

Learn OpenGL in Python -- Plan:
1. Basic workflow by tutorial
2. Textured mapping with reference from the github open source code
'''

import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

# Define the eight vertices of the rectangular cube box
vertices = (
           (1, -1, -1),
           (1, 1, -1),
           (-1, 1, -1),
           (-1, -1, -1),
           (1, -1, 1),
           (1, 1, 1),
           (-1, -1, 1),
           (-1, 1, 1)
           )

# Define the index of each vertices
edges = (
        (0,1),
        (0,3),
        (0,4),
        (2,1),
        (2,3),
        (2,7),
        (6,3),
        (6,4),
        (6,7),
        (5,1),
        (5,4),
        (5,7)
        )

# Define the surface
surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
            )

# A tuple of color
colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (0,1,0),
    (1,1,1),
    (0,1,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (1,1,1),
    (0,1,1),
    )

# ground_surfaces = (0, 1, 2, 3)
#
# ground_vertices = (
#     (-10, -0.1, 50),
#     (10, -0.1, 50),
#     (-10, -0.1, -300),
#     (10, -0.1, -300),
#
# )

# def Ground():
#     glBegin(GL_QUADS)
#
#     x = 0
#     for vertex in ground_vertices:
#         x += 1
#         glColor3fv((0, 1, 1))
#         glVertex3fv(vertex)
#
#     glEnd()

def set_vertices(max_distance, min_distance = -20, camera_x = 0, camera_y = 0):

    # Change set_vertices a bit by including the location of the "camera"
    camera_x = -1 * int(camera_x)
    camera_y = -1 * int(camera_y)

    # The position of each box
    x_value_change = random.randrange(-10,10)
    y_value_change = random.randrange(-10,10)
    z_value_change = random.randrange(-1*max_distance,-20)

    # Define a new vertices
    new_vertices = []

    # Define the vertice for each box
    for vert in vertices:

        new_vert = []

        new_x = vert[0] + x_value_change
        new_y = vert[1] + y_value_change
        new_z = vert[2] + z_value_change

        new_vert.append(new_x)
        new_vert.append(new_y)
        new_vert.append(new_z)

        new_vertices.append(new_vert)

    return new_vertices

# Generate the cube in OpenGL
def Cube(vertices):

    # Define the surface
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            # print(x)
            # x = 1
            # print(vertex)
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
    glEnd()

    # this notifies OpenGL that we're about to throw some code at it
    # GL_LINES tell opengl how to handle the code
    glBegin(GL_LINES) # Connect the lines

    for edge in edges:
        for vertex in edge:
         # It wants a pointer to an array of 3 float values (x,y,z) -- pass the value to opengl object
         # It's just another way to pass the vertex corrdinates to OpenGL,
         # it may be a little faster then using glVertex3f( v[ 0], v[ 1], v[ 2]);
         # glVertex3fv OpenGL function on the [vertex]
         glVertex3fv(vertices[vertex])
    glEnd()

def main():

    # Initialize the pygame engine
    pygame.init()

    # The size of the windows
    display = (800, 600)

    # Set the mode of the pygame engine
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    max_distance = 100

    # The perspective projection definition
    # (GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
    # zNear usually can be defined as 0.1 -- the distance between the projection plane and the eye
    # zFar can be defined based on the fact
    # fovy: The angle that you open the eye
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    # start further back
    glTranslatef(random.randrange(-5, 5), 0, -30)

    object_passed = False

    # Translation function -- move towards x y and z
    # Initialize the position of each box
    glTranslatef(0.0, 0.0, -5)
    # The position of each iteration for the moving objects
    x_move = 0
    y_move = 0
    max_distance = 100
    cube_dict = {}
    for x in range(20):
        cube_dict[x] = set_vertices(max_distance)

    # while True:
    while not object_passed:

        # Identify if the user is clicking close for the windows
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Try different event type
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-0.5, 0, 0)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(0.5, 0, 0)
                if event.key == pygame.K_UP:
                    glTranslatef(0, 1, 0)
                if event.key == pygame.K_DOWN:
                    glTranslatef(0, -1, 0)

            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     if event.button == 4:
            #         glTranslatef(0, 0, 1.0)
            #     if event.button == 5:
            #         glTranslatef(0, 0, -1.0)

        # Get our position in the world frame
        x = glGetDoublev(GL_MODELVIEW_MATRIX)
        camera_x = x[3][0]
        camera_y = x[3][1]
        camera_z = x[3][2]

        # Rotation function
        # The model will keep rotation defined by this vector
        # This is well defined by pygame
        glRotatef(1, 0, 0, 0)

        # The clear function -- clear the current parameters
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Define the ground
        # Ground()

        # glTranslatef(0, 0, 0.10)    # update the position in each state
        glTranslatef(x_move, y_move, .50)

        for each_cube in cube_dict:
            Cube(cube_dict[each_cube])

        for each_cube in cube_dict:
            if camera_z <= cube_dict[each_cube][0][2]:
                print("passed a cube")
                # delete_list.append(each_cube)
                new_max = int(-1 * (camera_z - max_distance))
                # Set all the boxs back to the starting position
                cube_dict[each_cube] = set_vertices(new_max, int(camera_z))

        # Cube()

        # Update the full display surface to the screen
        pygame.display.flip()

        # if camera_z <= 0:
        #     object_passed = True

        pygame.time.wait(1)

main()





