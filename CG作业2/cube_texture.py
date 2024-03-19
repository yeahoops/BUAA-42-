import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import glm # pip install PyGLM

VERTEX_SHADER = """
#version 430
 
    layout(location = 0) in vec4 position;
    layout(location = 1) in vec4 color;
    layout(location = 2) in vec2 texcoord;
    uniform mat4 MVP;

    out vec4 vt_color;
    out vec2 vt_texcoord;
    void main() {
        gl_Position = MVP * position;
        vt_color = color;
        vt_texcoord = texcoord;
    } 
"""

FRAGMENT_SHADER = """
#version 430
    in vec4 vt_color;
    in vec2 vt_texcoord;
    uniform sampler2D tex0;
    out vec4 outputColor;
    void main() {
        outputColor = vt_color * texture(tex0, vt_texcoord);
    }

"""

shaderProgram = None
VAO = None
NumVertices = 36

positions = [
                [-0.5, -0.5,  0.5, 1.0 ],
                [-0.5,  0.5,  0.5, 1.0 ],
                [0.5,  0.5,  0.5, 1.0 ],
                [0.5, -0.5,  0.5, 1.],
                [-0.5, -0.5, -0.5, 1.0 ],
                [-0.5,  0.5, -0.5, 1.0 ],
                [0.5,  0.5, -0.5, 1.0 ],
                [0.5, -0.5, -0.5, 1.0 ]
            ]

colors =    [
                [0.0, 0.0, 0.0, 1.0  ], #black
                [1.0, 0.0, 0.0, 1.0 ],  #red
                [1.0, 1.0, 0.0, 1.0 ],  #yellow
                [0.0, 1.0, 0.0, 1.0 ],  #green
                [0.0, 0.0, 1.0, 1.0 ],  #blue
                [1.0, 0.0, 1.0, 1.0 ],  #magenta
                [1.0, 1.0, 1.0, 1.0 ],  #white
                [0.0, 1.0, 1.0, 1.0 ]   #cyan
            ]
vColors = [ [] for i in range(NumVertices) ]
vPositions = [ [] for i in range(NumVertices) ]
vTexCoords = [ [] for i in range(NumVertices)]
index = 0
def quad( a, b, c, d):
    global index
    vColors[index], vPositions[index], vTexCoords[index], index = colors[a], positions[a], [0.0, 0.0], index+1
    vColors[index], vPositions[index], vTexCoords[index], index = colors[b], positions[b], [1.0, 0.0], index+1
    vColors[index], vPositions[index], vTexCoords[index], index = colors[c], positions[c], [1.0, 1.0], index+1
    vColors[index], vPositions[index], vTexCoords[index], index = colors[a], positions[a], [0.0, 0.0], index+1
    vColors[index], vPositions[index], vTexCoords[index], index = colors[c], positions[c], [1.0, 1.0], index+1
    vColors[index], vPositions[index], vTexCoords[index], index = colors[d], positions[d], [0.0, 1.0], index+1

def colorcube():

    quad(1, 0, 3, 2)
    quad(2, 3, 7, 6)
    quad(3, 0, 4, 7)
    quad(6, 5, 1, 2)
    quad(4, 5, 6, 7)
    quad(5, 4, 0, 1)

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
 
    colorcube()
    points = np.array(vPositions, np.float32)
    colors = np.array(vColors, np.float32)
    texcoords = np.array(vTexCoords, np.float32)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes+colors.nbytes+texcoords.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes, colors.nbytes, colors)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes+colors.nbytes, texcoords.nbytes, texcoords)

    #position = glGetAttribLocation(shaderProgram, 'position')
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(points.nbytes))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(points.nbytes+colors.nbytes))
    glEnableVertexAttribArray(2)

    tex = glGenTextures(1)
    img = checkborad_pattern()
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )



def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(90.0),float(width)/float(height),0.1,10.0)
    view = glm.lookAt(glm.vec3(1.0,1.0,1.0), glm.vec3(0,0,0),glm.vec3(0,1,0))
    
    model =  glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(45), glm.vec3(0, 0, 1))
    
    mvp = proj * view * model
    
    return mvp



def render():
    global shaderProgram
    global VAO

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_MULTISAMPLE)

    glUseProgram(shaderProgram)

    mvp_loc = glGetUniformLocation(shaderProgram,"MVP")
    mvp_mat = calc_mvp(640, 480)
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp_mat))
    
    glActiveTexture(GL_TEXTURE0)
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0)

    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, NumVertices)
   
    glUseProgram(0)

    glutSwapBuffers()


def main():
   
    glutInit([])
    #glutSetOption(GLUT_MULTISAMPLE, 8)
    #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE )
    glutInitWindowSize(640, 480)
    glutCreateWindow("pyopengl with glut")
    initliaze()
    glutDisplayFunc(render)

    glutMainLoop()
 
 
if __name__ == '__main__':
    main()

