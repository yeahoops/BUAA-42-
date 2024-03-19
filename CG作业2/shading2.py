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
    layout(location = 1) in vec4 normal;
    uniform mat4 Model;
    uniform mat4 View;
    uniform mat4 Projection;

    out vec3 Normal;
    out vec3 Position;
    void main() {
        gl_Position = Projection * View * Model * position;
        Normal = mat3(transpose(inverse(Model))) * normal.xyz;
        Position = vec3(Model * position);
    }
"""

FRAGMENT_SHADER = """
#version 430

    in vec3 Normal;
    in vec3 Position;
    uniform vec3 cameraPos;
    uniform samplerCube skybox;
    out vec4 outputcolor;
    void main() {
        vec3 I = normalize(cameraPos - Position);
        vec3 R = reflect(I, normalize(Normal));
        outputcolor = texture(skybox, R);
    }
"""

SKYBOX_VERTEX_SHADER = """
#version 430

    layout(location = 0) in vec4 position;
    uniform mat4 MVP;
    out vec4 tex_position;
    void main() {
        gl_Position = MVP * position;
        tex_position = position;
    }
"""

SKYBOX_FRAGMENT_SHADER = """
#version 430

    uniform samplerCube skybox;
    in vec4 tex_position;
    out vec4 outputcolor;
    void main() {
        outputcolor = texture(skybox, tex_position.xyz);
    }
"""

cubeVAO = None
skyboxVAO = None
cubeVBO = None
skyboxVBO = None
NumVertices = 36
skyboxTexture = None

time_count = 0

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

spositions = [
                [-1, -1,  1, 1.0 ],
                [-1,  1,  1, 1.0 ],
                [1,  1,  1, 1.0 ],
                [1, -1,  1, 1.],
                [-1, -1, -1, 1.0 ],
                [-1,  1, -1, 1.0 ],
                [1,  1, -1, 1.0 ],
                [1, -1, -1, 1.0 ]
            ]



cubePositions = [ [] for i in range(NumVertices) ]
cubeNormals = [ [] for i in range(NumVertices) ]
skyboxPositions = [ [] for i in range(NumVertices) ]


index = 0
def quad(a, b, c, d):
    global index
    pos_a, pos_b, pos_c, pos_d = np.array(positions[a][0:3]), np.array(positions[b][0:3]),  np.array(positions[c][0:3]),  np.array(positions[d][0:3])
    normal = (-np.cross(pos_b-pos_a, pos_b-pos_c)).tolist()
    normal.append(1.0)
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[a], normal, spositions[a], index+1
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[b], normal, spositions[b], index+1
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[c], normal, spositions[c], index+1

    normal = (-np.cross(pos_d-pos_c, pos_d-pos_a)).tolist()
    normal.append(1.0)
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[a], normal, spositions[a], index+1
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[c], normal, spositions[c], index+1
    cubePositions[index], cubeNormals[index], skyboxPositions[index], index = positions[d], normal, spositions[d], index+1

def colorcube():

    quad(1, 0, 3, 2)
    quad(2, 3, 7, 6)
    quad(3, 0, 4, 7)
    quad(6, 5, 1, 2)
    quad(4, 5, 6, 7)
    quad(5, 4, 0, 1)

def initliaze():
    global cubevertexshader
    global cubefragmentshader
    global skyboxvertexshader
    global skyboxfragmentshader
    global cubeshaderProgram
    global skyboxshaderProgram
    global cubeVAO
    global cubeVBO
    global skyboxVAO
    global skyboxVBO
    global skyboxTexture


    cubevertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    cubefragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    cubeshaderProgram = shaders.compileProgram(cubevertexshader, cubefragmentshader)
    
    skyboxvertexshader = shaders.compileShader(SKYBOX_VERTEX_SHADER, GL_VERTEX_SHADER)
    skyboxfragmentshader = shaders.compileShader(SKYBOX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    skyboxshaderProgram = shaders.compileProgram(skyboxvertexshader, skyboxfragmentshader)
    
    colorcube()
    points = np.array(cubePositions, np.float32)
    normals = np.array(cubeNormals, np.float32)
    skyboxPoints = np.array(skyboxPositions, np.float32)


    cubeVAO = glGenVertexArrays(1)
    glBindVertexArray(cubeVAO)
    cubeVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes+normals.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes, normals.nbytes, normals)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16 , None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16 , ctypes.c_void_p(points.nbytes))
    glEnableVertexAttribArray(1)


    skyboxVAO = glGenVertexArrays(1)
    glBindVertexArray(skyboxVAO)
    skyboxVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO)
    glBufferData(GL_ARRAY_BUFFER, skyboxPoints.nbytes, skyboxPoints, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None)
    glEnableVertexAttribArray(0)


    skyboxTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture)
    name = ['right', 'left', 'top', 'bottom', 'front', 'back']
    for i in range(6):
        img = np.array(Image.open(name[i] + '.png'))
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, img.shape[1], img.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, img)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT)

def calc_cube_mvp(width, height):
    proj = glm.perspective(glm.radians(90.0),float(width)/float(height),0.1,10.0)
    view = glm.lookAt(glm.vec3(1.0,1.0,1.0), glm.vec3(0,0,0),glm.vec3(0,1,0))

    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(time_count * 5), glm.vec3(0, 1, 1))

    return model, view, proj

def calc_skybox_mvp(width, height):
    proj = glm.perspective(glm.radians(90.0),float(width)/float(height),0.1,10.0)
    view = glm.lookAt(glm.vec3(0.75,0.75,0.75), glm.vec3(0,0,0),glm.vec3(0,1,0))

    model = glm.mat4(1.0)
    model = glm.translate(model, glm.vec3(0.75, 0.75, 0.75))


    return model, view, proj




def render():

    global cube

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    m, v, p = calc_skybox_mvp(640, 480)

    glDepthMask(GL_FALSE)
    glUseProgram(skyboxshaderProgram)
    glBindVertexArray(skyboxVAO)
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture)
    glUniformMatrix4fv(glGetUniformLocation(skyboxshaderProgram, "MVP"), 1, GL_FALSE, glm.value_ptr(p*v*m))
    glUniform1i(glGetUniformLocation(skyboxshaderProgram, "skybox"), 0)
    glDrawArrays(GL_TRIANGLES, 0, NumVertices)







    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    #glEnable(GL_MULTISAMPLE)


    glUseProgram(cubeshaderProgram)
    m, v, p = calc_cube_mvp(640, 480)
    glUniformMatrix4fv(glGetUniformLocation(cubeshaderProgram, "Model"), 1, GL_FALSE, glm.value_ptr(m))
    glUniformMatrix4fv(glGetUniformLocation(cubeshaderProgram, "View"), 1, GL_FALSE, glm.value_ptr(v))
    glUniformMatrix4fv(glGetUniformLocation(cubeshaderProgram, "Projection"), 1, GL_FALSE, glm.value_ptr(p))
    glUniform1i(glGetUniformLocation(cubeshaderProgram, "skybox"), 0)
    glBindVertexArray(cubeVAO)
    glDrawArrays(GL_TRIANGLES, 0, NumVertices)

    glUseProgram(0)

    glutSwapBuffers()

def animate(value):
    global time_count

    glutPostRedisplay()

    glutTimerFunc(50, animate, 0)

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


