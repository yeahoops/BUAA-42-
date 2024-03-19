import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import glm # pip install PyGLM

VERTEX_SHADER = """
#version 430
 
    layout(location = 0) in vec4 position;
    layout(location = 1) in vec2 texcoord;
    uniform mat4 MVP;

    out vec2 vt_texcoord;
    void main() {
        gl_Position = MVP * position;
        vt_texcoord = texcoord;
    } 
"""

FRAGMENT_SHADER = """
#version 430
    in vec2 vt_texcoord;
    uniform sampler2D tex0;
    out vec4 outputColor;
    void main() {
        outputColor = texture(tex0, vt_texcoord);
    }
"""

shaderProgram = None
VAO = None

time_count = 0

def tetrahedron(): #四面体
    pointList = [
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, -1.0, -1.0, 1.0],
                    [-1.0, -1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, 1.0]
                ] #4个顶点坐标
    
    indexList = [
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 3, 1],
                    [1, 2, 3]
                ] #4个三角形，每个三角形对应的三个顶点的索引
    

    texcoordList = [
                       [0.5, 0.5],
                       [0.25, 0.25],
                       [0.75, 0.25],
                       [0.5, 0.75]
                   ] #4个顶点对应的纹理坐标
    
    points, texcoords = [], []
    for i in range(4):
        points.append(pointList[indexList[i][0]])
        points.append(pointList[indexList[i][1]])
        points.append(pointList[indexList[i][2]]) #三角形三个顶点的坐标

        texcoords.append(texcoordList[indexList[i][0]])
        texcoords.append(texcoordList[indexList[i][1]])
        texcoords.append(texcoordList[indexList[i][2]]) #三角形三个顶点的纹理坐标

    return np.array(points, np.float32), np.array(texcoords, np.float32)

def checkborad_pattern():
    img = np.zeros((64, 64, 3), 'uint8')
    for i in range(64):
        for j in range(64):
            c = ((i & 0x8 == 0) ^ (j & 0x8 == 0)) * 255
            img[i][j][0] = c
            img[i][j][1] = c
            img[i][j][2] = c
    
    return img

def initliaze():
    global VERTEXT_SHADER
    global FRAGMEN_SHADER
    global shaderProgram
    global VAO
 
 
    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
 
    
 
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)

    points, texcoords = tetrahedron()
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes+texcoords.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes, texcoords.nbytes, texcoords)

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(points.nbytes))
    glEnableVertexAttribArray(1)

    tex = glGenTextures(1)
    img = np.array(Image.open('earthmap.jpg'))
    print(img.shape, img.dtype)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )



def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0),float(width)/float(height),0.1,20.0)
    view = glm.lookAt(glm.vec3(3.0,0.0,2.0), glm.vec3(0,0,0),glm.vec3(0,1,0))
    
    model =  glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(time_count * 5), glm.vec3(0, 1, 0))
    
    mvp = proj * view * model
    
    return mvp



def render():
    global shaderProgram
    global VAO

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    #glEnable(GL_MULTISAMPLE)

    glUseProgram(shaderProgram)

    mvp_loc = glGetUniformLocation(shaderProgram,"MVP")
    mvp_mat = calc_mvp(640, 480)
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp_mat))
    
    glActiveTexture(GL_TEXTURE0)
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0)

    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 12) #VAO里4个三角形，每个三角形3个顶点，总共12个顶点
    
   
    glUseProgram(0)

    glutSwapBuffers()

def animate(value):
    global time_count

    glutPostRedisplay()

    glutTimerFunc(200, animate, 0)
    
    time_count = time_count+1.0

def main():
   
    glutInit([])
    #glutSetOption(GLUT_MULTISAMPLE, 8)
    #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE )
    glutInitWindowSize(640, 480)
    glutCreateWindow("pyopengl with glut")
    initliaze()
    glutDisplayFunc(render)

    glutTimerFunc(200, animate, 0)
    glutMainLoop()
 
 
if __name__ == '__main__':
    main()

