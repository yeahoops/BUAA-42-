import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import glm  # pip install PyGLM

VERTEX_SHADER = """
#version 430
 
layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 norm;
uniform mat4 MVP;

out vec2 vt_texcoord;
out vec4 Position;
out vec3 Norm;

void main() {
    gl_Position = MVP * position;
    Position = position;
    Norm = norm;
    vt_texcoord = texcoord;

}
"""

FRAGMENT_SHADER = """
#version 430
uniform sampler2D tex0;
uniform sampler2D depth;

uniform mat4 M; // 法线变换矩阵
uniform vec3 lightPos;    // 光源在世界坐标下的位置
uniform vec3 lightColor;  // 光源的颜色
uniform vec3 viewPos;     // 视角在世界坐标下的位置

in vec2 vt_texcoord;
in vec4 Position;
in vec3 Norm;

out vec4 outputColor;

void main() {
    vec3 norm = mat3(transpose(inverse(M))) * Norm;

    vec2 left_Coord = vec2(vt_texcoord.x - 0.001, vt_texcoord.y);
    vec2 right_Coord = vec2(vt_texcoord.x + 0.001, vt_texcoord.y);
    vec2 up_Coord = vec2(vt_texcoord.x, vt_texcoord.y + 0.001);
    vec2 down_Coord = vec2(vt_texcoord.x, vt_texcoord.y - 0.001);

    vec3 U = normalize(vec3(norm.y, norm.x, 0.0));
    vec3 V = normalize(cross(norm, U));
    vec3 u_bias = (texture(depth,left_Coord).r - texture(depth,right_Coord).r)*U;
    vec3 v_bias = (texture(depth,down_Coord).r - texture(depth,up_Coord).r)*V;
    vec3 new_norm = norm + u_bias + v_bias;
    

    // 环境光
    float ambientStrength = 0.05;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    float diffuseStrength = 1;
    vec3 dd = lightPos - (M * Position).xyz;
    vec3 lightDir = normalize(dd);
    vec3 diffuse = lightColor * max(dot(new_norm, lightDir), 0.0) * diffuseStrength;

    // 镜面反射
    float specularStrength = 2;
    vec3 viewDir = normalize(viewPos - (M * Position).xyz); 
    vec3 reflectDir = reflect(-lightDir, new_norm); 
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32); 
    vec3 specular = lightColor * spec * specularStrength;

    vec4 fliter = vec4(ambient + diffuse + specular, 1);
    vec4 kd = texture(tex0, vt_texcoord);

    outputColor = kd * fliter;

}
"""

shaderProgram = None
VAO = None

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

    return np.array(vertices, np.float32), np.array(texcoords, np.float32), np.array(normals, np.float32), np.array(indices, np.uint32)

def initialize():
    global VERTEX_SHADER
    global FRAGMENT_SHADER
    global shaderProgram
    global VAO

    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)

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

    tex = glGenTextures(1)
    img = np.array(Image.open('earthmap.jpg'))
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )
    
    bump_map = glGenTextures(1)
    bump_map_img = np.array(Image.open("Earth-Bump-Map.jpg"))
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, bump_map)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, bump_map_img.shape[1], bump_map_img.shape[0], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, bump_map_img)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )

def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0), float(width) / float(height), 0.1, 20.0)
    view = glm.lookAt(glm.vec3(2.0, 2.0, 2.0), glm.vec3(0, 0, 0), glm.vec3(0, 0, 1))

    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(time_count * 5), glm.vec3(0, 0, -1))

    mvp = proj * view * model

    return model, mvp

def render():
    global shaderProgram
    global VAO

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

    glUseProgram(shaderProgram)

    mvp_loc = glGetUniformLocation(shaderProgram, "MVP")
    m_loc = glGetUniformLocation(shaderProgram, "M")
    m_mat, mvp_mat = calc_mvp(640, 480)
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp_mat))
    glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m_mat))

    lightPos = np.array([4, -2, 5], np.float32)
    lightColor = np.array([1.0, 1.0, 1.0], np.float32)
    viewpos = np.array([2.0, 2.0, 2.0], np.float32)
    glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos[0], lightPos[1], lightPos[2])
    glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), lightColor[0], lightColor[1], lightColor[2])
    glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"),  viewpos[0], viewpos[1], lightColor[2])
    glActiveTexture(GL_TEXTURE0)
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0)
    glActiveTexture(GL_TEXTURE1)
    glUniform1i(glGetUniformLocation(shaderProgram, "depth"), 1)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6 * 50 * 50, GL_UNSIGNED_INT, None)

    glUseProgram(0)

    glutSwapBuffers()

def animate(value):
    global time_count

    glutPostRedisplay()

    glutTimerFunc(200, animate, 0)

    time_count = time_count + 1.0

def main():
    glutInit([])
    glutInitWindowSize(640, 480)
    glutCreateWindow("Sphere with Texture Mapping")
    initialize()
    glutDisplayFunc(render)
    glutTimerFunc(200, animate, 0)
    glutMainLoop()

if __name__ == '__main__':
    main()
