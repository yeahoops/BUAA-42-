import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo
from OpenGL.GL import shaders

from shader import Shader

import glm # pip install PyGLM

VERTEX_SHADER = """
#version 430

    layout(location = 0) in vec4 position;
    layout(location = 1) in vec2 texcoord;
    layout(location = 2) in vec3 normal;
    uniform mat4 MVP;
    uniform mat4 M;
    uniform mat4 LS;

    out vec3 Position;
    out vec2 TexCoord;
    out vec3 Normal;
    out vec4 lightPosition;

    void main()
    {
        gl_Position = MVP * position;
        Position = vec3(M * position);
        TexCoord = texcoord;
        Normal = normalize(mat3(M) * normal);
        lightPosition = LS * position;
    }
"""

FRAGMENT_SHADER = """
#version 430
    in vec3 Position;
    in vec2 TexCoord;
    in vec3 Normal;
    in vec4 lightPosition;

    uniform sampler2D tex;
    uniform sampler2D depth;
    uniform vec3 lightPos;
    uniform vec3 lightColor;
    uniform float diffuseStrength;

    out vec4 outputColor;
    void main()
    {
        vec3 lightDir = -normalize(Position - lightPos);
        float diff = max(0.0, dot(lightDir, Normal));
        vec3 diffuse = diff * lightColor * diffuseStrength;

        vec3 ambient = lightColor * 0.15;

        vec3 depthTexCoord = lightPosition.xyz / lightPosition.w;
        depthTexCoord = depthTexCoord * 0.5 + 0.5;
        float dep = texture(depth, depthTexCoord.xy).r;
        float shadow = depthTexCoord.z < dep + 0.005 ? 1.0 : 0;

        outputColor = texture(tex, TexCoord) * vec4(diffuse * shadow + ambient, 1);
    }
"""

SHADOW_VERTEX_SHADER = """
#version 430

    layout(location = 0) in vec4 position;
    out vec4 Position;
    uniform mat4 MVP;
    void main()
    {
        gl_Position = MVP * position;
        Position = MVP * position;
    }
"""

SHADOW_FRAGMENT_SHADER = """
#version 430
    in vec4 Position;

    out vec4 outputColor;
    void main()
    {
        outputColor = vec4(vec3(Position.z / Position.w), 1.0);
    }
"""

shader = None
shadowShader = None
VAO = None
VBO = None
planeVAO = None
SHADOW_WIDTH = 1920
SHADOW_HEIGHT = 1080
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
depthMapFBO = None

time_count = 0

def generate_sphere(radius, num_segments):
    vertices = []
    normals = []
    texcoords = []


    for i in range(num_segments + 1):
        for j in range(num_segments + 1):
            theta = i * np.pi / num_segments
            phi = j * 2 * np.pi / num_segments

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            vertices.append([x, y, z, 1.0])

            norm = np.array([x, y, z])
            norm = norm / np.linalg.norm(norm)
            normals.append(norm)

            s = j / num_segments
            t = i / num_segments
            texcoords.append([s, t])


    indices = []
    for i in range(num_segments):
        for j in range(num_segments):
            v0 = i * (num_segments + 1) + j
            v1 = v0 + 1
            v2 = (i + 1) * (num_segments + 1) + j
            v3 = v2 + 1

            indices.extend([v0, v1, v2, v1, v3, v2])

    return np.array(vertices, np.float32), np.array(texcoords, np.float32), np.array(normals, np.float32),  np.array(indices, np.uint32)

def initialize():
    global VERTEXT_SHADER
    global FRAGMEN_SHADER
    global SHADOW_WIDTH
    global SHADOW_HEIGHT
    global depthMapFBO
    global shader
    global shadowShader
    global VAO
    global planeVAO

    shader = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
    shadowShader = Shader(SHADOW_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER)

    vertices, texcoords, normals, indices = generate_sphere(radius=1.0, num_segments=50)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(4)

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
    glBufferData(GL_ARRAY_BUFFER, texcoords.nbytes, texcoords, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, VBO[2])
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[3])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    planeV = np.array([[4, 2, -1, 1], [-4, 2, -1, 1], [0, -3, -1, 1],
                       [-3, -3, 0, 1], [3, 1, 0, 1], [3, -3, 0, 1]], np.float32)
    planeT = np.array([[0, 0], [1, 1], [0, 1],
                       [0, 0], [1, 1], [1, 0]], np.float32)
    planeN = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1],
                       [0, 0, 1], [0, 0, 1], [0, 0, 1]], np.float32)

    planeVAO = glGenVertexArrays(1)
    glBindVertexArray(planeVAO)

    planeVBO = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO[0])
    glBufferData(GL_ARRAY_BUFFER, planeV.nbytes, planeV, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, planeVBO[1])
    glBufferData(GL_ARRAY_BUFFER, planeT.nbytes, planeT, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, planeVBO[2])
    glBufferData(GL_ARRAY_BUFFER, planeN.nbytes, planeN, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)

    tex = glGenTextures(1)
    img = np.array(Image.open('earthmap.jpg'))

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )

    planeTex = glGenTextures(1)
    planeImg = np.array(Image.open('plane.jpg'))

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, planeTex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, planeImg.shape[1], planeImg.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, planeImg)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )

    glActiveTexture(GL_TEXTURE2)
    depthMapFBO = glGenFramebuffers(1)
    depthMap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0),float(width)/float(height),0.1,20.0)
    view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0,0,0),glm.vec3(0,1,0))
    model =  glm.mat4(1.0)
    #model = glm.rotate(model, glm.radians(time_count * 5), glm.vec3(0, 1, 0))
    return model, view, proj

def calc_light_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0),float(width)/float(height),0.1,20.0)
    view = glm.lookAt(glm.vec3(-2, -2, 3), glm.vec3(0,0,0),glm.vec3(0,1,0))
    model =  glm.mat4(1.0)
    #model = glm.rotate(model, glm.radians(time_count * 5), glm.vec3(0, 1, 0))
    return model, view, proj


def render():
    global shaderProgram
    global VAO

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    #glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClear(GL_DEPTH_BUFFER_BIT)
    lm, lv, lp = calc_light_mvp(SHADOW_WIDTH, SHADOW_HEIGHT)


    shadowShader.use()
    shadowShader.setMatrix4fv("MVP", lp * lv * lm)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6 * 50 * 50, GL_UNSIGNED_INT, None)
    glBindVertexArray(planeVAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)

#--------------------------------------------------------------------

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#
    shader.use()
    m, v, p = calc_mvp(SCREEN_WIDTH, SCREEN_HEIGHT)
    shader.setMatrix4fv("MVP", p * v * m)
    shader.setMatrix4fv("MV", v * m)
    shader.setMatrix4fv("M", m)
    shader.setMatrix4fv("LS", lp * lv)
    lightPos = p * v * glm.vec3(-2, -2, 3)
    shader.set3fvFromList("lightPos",[lightPos.x, lightPos.y, lightPos.z])
    shader.set3fvFromList("lightColor",[1, 1, 1])
    shader.setFloat("diffuseStrength", 1.5)
    shader.setInt("depth", 2)
#
    shader.setInt("tex", 0)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6 * 50 * 50, GL_UNSIGNED_INT, None)
#
    shader.setInt("tex", 1)
    glBindVertexArray(planeVAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)

    glUseProgram(0)

    glutSwapBuffers()

def animate(value):
    global time_count

    glutPostRedisplay()

    glutTimerFunc(90, animate, 0)

    time_count = time_count+1.0

def main():

    glutInit([])
    #glutSetOption(GLUT_MULTISAMPLE, 8)
    #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE )
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT)
    glutCreateWindow("pyopengl with glut")
    initialize()
    glutDisplayFunc(render)
    glutTimerFunc(20, animate, 0)
    glutMainLoop()


if __name__ == '__main__':
    main()
