#!/usr/bin/python
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

class Window(object):
    """An abstract GLUT window."""

    def __init__(self, source=None, title="Untitled Window", width=500, height=500, ortho=None):
        """Constructs a window with the given title and dimensions. Source is the original redbook file."""
        self.source = source
        self.ortho = ortho
        self.width = width
        self.height = height
        self.keybindings = {chr(27): exit}
        glutInit()
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(title)
        # Just request them all and don't worry about it.
        #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glEnable(GL_MULTISAMPLE)
        glutSetOption(GLUT_MULTISAMPLE, 4)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
        
        glClearColor(0, 0, 0, 0)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutDisplayFunc(self.display)
        glutMouseFunc(self.mouse)
        glShadeModel(GL_FLAT)
        
        glEnable (GL_LINE_SMOOTH)
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE)


    def keyboard(self, key, mouseX, mouseY):
        '''Call the code mapped to the pressed key.'''
        self.keybindings.get(key, noop)()
        glutPostRedisplay()

    def mouse(self, button, state, x, y):
        '''Handle mouse clicking.'''
        if button == GLUT_LEFT_BUTTON:
            self.mouseLeftClick(x, y)
        elif button == GLUT_MIDDLE_BUTTON:
            self.mouseMiddleClick(x, y)
        elif button == GLUT_RIGHT_BUTTON:
            self.mouseRightClick(x, y)
        else:
            raise ValueError(button)
        glutPostRedisplay()

    def mouseLeftClick(self, x, y):
        pass

    def mouseMiddleClick(self, x, y):
        pass

    def mouseRightClick(self, x, y):
        pass

    def reshape(self, width, height):
        '''Recalculate the clipping window the GLUT window is resized.'''
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = float(self.height) / float(self.width)
        # ortho is the scaling factor for the orthogonal projection
        if self.ortho:
            if aspect >= 1:
                glOrtho(-2.0, 2.0, -2.0 * aspect, 2.0 * aspect, -2.0, 2.0)
            else:
                glOrtho(-2.0 / aspect, 2.0 / aspect, -2.0, 2.0, -2.0, 2.0)
        else:
            gluPerspective(30, 1.0 / aspect, 1, 20)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def display(self):
        '''Children implement this to define their rendering behavior.'''
        raise NotImplementedError

    @staticmethod
    def run():
        """Start up the main loop."""
        glutMainLoop()


class BezierCurve(Window):
    """Use evaluators to draw a Bezier curve."""

    def __init__(self):
        """Constructor"""
        super(BezierCurve, self).__init__("bezcurve.c", "Bezier Curve", 500, 500, True)
        self.controlPoints = (
            (-3 / 5.0, -4 / 5.0, 0), (-2 / 5.0, 4 / 5.0, 0), (2 / 5.0, -4 / 5.0, 0), (3 / 5.0, 4 / 5.0, 0))
        glClearColor(0, 0, 0, 0)
        glShadeModel(GL_FLAT)
        glMap1f(GL_MAP1_VERTEX_3, 0, 1, self.controlPoints)
        glEnable(GL_MAP1_VERTEX_3)

    def display(self):
        """Display the control points as dots."""
        glClear(GL_COLOR_BUFFER_BIT)

        # 下面一段请在1.2作业中注释
        # glBegin(GL_LINE_STRIP)
        # for i in range(31):
        #     glEvalCoord1f(float(i) / 30)
        # glEnd()

        glPointSize(5)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 0)
        for point in self.controlPoints:
            glVertex3fv(point)
        glEnd()

         # Draw the Bezier curve using manual calculation
        glColor3f(1, 1, 1)
        glBegin(GL_LINE_STRIP)

        # Use De Casteljau's algorithm to compute points on the Bezier curve
        num_segments = 100  # Number of segments to approximate the curve
        for i in range(num_segments + 1):
            t = i / num_segments
            curve_point = self.compute_bezier_point(t)
            glVertex3fv(curve_point)

        glEnd()
        glFlush()

    def compute_bezier_point(self, t):
        """Compute a point on the Bezier curve at parameter t using De Casteljau's algorithm."""
        n = len(self.controlPoints) - 1
        temp_points = [point for point in self.controlPoints]  # Copy control points

        for k in range(1, n + 1):
            for i in range(n - k + 1):
                temp_points[i] = ((1 - t) * temp_points[i][0] + t * temp_points[i + 1][0],
                                    (1 - t) * temp_points[i][1] + t * temp_points[i + 1][1],
                                    (1 - t) * temp_points[i][2] + t * temp_points[i + 1][2])

        return temp_points[0]

        # 1.2

        
class BezierCircle(Window):
    """Use evaluators to draw a Bezier curve."""

    def __init__(self):
        """Constructor"""
        super(BezierCircle, self).__init__("bezcurve.c", "Bezier Curve", 500, 500, True)
        t = 4 / 3 * (np.sqrt(2) - 1) # magic number
        self.controlPoints1 = ((0, 1, 0), (t, 1, 0), (1, t, 0), (1, 0, 0))
        self.controlPoints2 = ((0, 1, 0), (-t, 1, 0), (-1, t, 0), (-1, 0, 0))
        self.controlPoints3 = ((0, -1, 0), (t, -1, 0), (1, -t, 0), (1, 0, 0))
        self.controlPoints4 = ((0, -1, 0), (-t, -1, 0), (-1, -t, 0), (-1, 0, 0))
        glClearColor(0, 0, 0, 0)
        glShadeModel(GL_FLAT)

    def display(self):
        """Display the control points as dots."""
        glClear(GL_COLOR_BUFFER_BIT)

        # 绘制四段曲线
        self.draw_bezier_curve(self.controlPoints1)
        self.draw_bezier_curve(self.controlPoints2)
        self.draw_bezier_curve(self.controlPoints3)
        self.draw_bezier_curve(self.controlPoints4)

    def draw_bezier_curve(self, controlPoints):
        """Draw a Bezier curve for given control points."""
        glMap1f(GL_MAP1_VERTEX_3, 0, 1, controlPoints)
        glEnable(GL_MAP1_VERTEX_3)
        glPointSize(5)
        glColor3f(1, 1, 0)
        glBegin(GL_POINTS)
        for point in controlPoints:
            glVertex3fv(point)
        glEnd()
        
        glColor3f(1, 1, 1)
        glBegin(GL_LINE_STRIP)
        for i in range(31):
            glEvalCoord1f(float(i) / 30)
        glEnd()
        glFlush()

class BezierSurface(Window):
    """Use evaluators to draw a Bezier Surface."""

    def __init__(self):
        """Constructor"""
        super(BezierSurface, self).__init__("bezsurface.c", "Bezier Surface", 500, 500, True)
        self.controlPoints = [
            [
                [-1.5, -1.5, 2.0],
                [-0.5, -1.5, 2.0],
                [0.5, -1.5, -1.0],
                [1.5, -1.5, 2.0]
            ],
            [
                [-1.5, -0.5, 1.0],
                [-0.5, 1.5, 2.0],
                [0.5, 0.5, 1.0],
                [1.5, -0.5, -1.0]
            ],
            [
                [-1.5, 0.5, 2.0],
                [-0.5, 0.5, 1.0],
                [0.5, 0.5, 3.0],
                [1.5, -1.5, 1.5]
            ],
            [
                [-1.5, 1.5, -2.0],
                [-0.5, 1.5, -2.0],
                [0.5, 0.5, 1.0],
                [1.5, 1.5, -1.0]
            ]
        ]
        glClearColor(0, 0, 0, 0)
        glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1, self.controlPoints)
        glMapGrid2f(20, 0.0, 1.0, 20, 0.0, 1.0)
        glEnable(GL_MAP2_VERTEX_3)
        glEnable(GL_DEPTH_TEST)

        # 下面一段请在作业2.2时注释掉
        # ambient = [0.2, 0.3, 0.4, 1.0]
        # position = [0.0, 1.0, 3.0, 1.0]
        # mat_diffuse = [0.1, 0.3, 0.4, 1.0]
        # mat_specular = [1.0, 1.0, 1.0, 1.0]
        # mat_shininess = [50.0]
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        # glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        # glLightfv(GL_LIGHT0, GL_POSITION, position)
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        # glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        # glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)
        # glShadeModel(GL_SMOOTH)

    def display(self):
        """Display the control points as dots."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(0.0, 1.0, 0.0)
        #glRotatef(45.0,45.0, 45.0, 1.0)
        #glEvalMesh2(GL_FILL, 0, 20, 0, 20)

        glBegin(GL_POINTS)
        num_segments = 30  # Number of segments to approximate the curve
        for i in range(num_segments + 1):
            for j in range(num_segments + 1):
                u = i / num_segments
                v = j / num_segments
                curve_point = self.compute_bezier_point(u, v)
                glVertex3fv(curve_point)
        glEnd()
        glFlush()

        # 2.2

        glutSwapBuffers()

    def compute_bezier_point(self, u, v):
        """Compute a point on the Bezier curve at parameter t using De Casteljau's algorithm."""
        n = len(self.controlPoints) - 1
        m = len(self.controlPoints[0]) - 1
        temp_points = [[point for point in row] for row in self.controlPoints]  # Copy control points
        
        for t in range(n + 1):
            for l in range(1, m + 1):
                for j in range(m - l + 1):
                    temp_points[t][j] = (
                        (1 - u) * temp_points[t][j][0] + u * temp_points[t][j + 1][0],
                        (1 - u) * temp_points[t][j][1] + u * temp_points[t][j + 1][1],
                        (1 - u) * temp_points[t][j][2] + u * temp_points[t][j + 1][2]
                    )

        for k in range(1, n + 1):
            for i in range(n - k + 1):
                temp_points[i][j] = (
                        (1 - v) * temp_points[i][0][0] + v * temp_points[i + 1][0][0],
                        (1 - v) * temp_points[i][0][1] + v * temp_points[i + 1][0][1],
                        (1 - v) * temp_points[i][0][2] + v * temp_points[i + 1][0][2]
                    )

        return temp_points[0][0]
if __name__ == '__main__':
    #BezierCurve().run()
    BezierCircle().run()
    #BezierSurface().run()
