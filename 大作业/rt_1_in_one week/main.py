import pygame
import math
import random
import threading 
from camera import Camera
from scene import Scene
# globals
BLACK = (0, 0, 0)
RED = (255, 0, 0)

vec3 = pygame.Vector3
rand01 = random.random

###################################################################################################
# Main application window and process
class Application:
    # ctor
    def __init__(self, size = (800, 600), caption = "PyGame window"):

        # state attributes
        self.__run = True

        # PyGame initialization
        pygame.init()
        self.__init_surface(size)
        pygame.display.set_caption(caption)
        
        self.__clock = pygame.time.Clock()

    # dtor
    def __del__(self):
        pygame.quit()


    # set the size of the application window
    @property
    def size(self):
        return self.__surface.get_size()

    # get window surface
    @property
    def surface(self):
        return self.__surface

    # get and set application 
    @property
    def image(self):
        return self.__image
    @image.setter
    def image(self, image):
        self.__image = image

    # main loop of the application
    def run(self, render, samples_per_pixel = 100, samples_update_rate = 1, capture_interval_s = 0):
        size = self.__surface.get_size()
        render.start(size, samples_per_pixel, samples_update_rate)
        finished = False
        start_time = None
        capture_i = 0
        while self.__run:
            self.__clock.tick(60)
            self.__handle_events()
            current_time = pygame.time.get_ticks()
            if start_time == None:
                start_time = current_time + 1000
            if not self.__run:
                render.stop()
            elif size != self.__surface.get_size():
                size = self.__surface.get_size()
                render.stop()
                render.start(size, samples_per_pixel, samples_update_rate)   
            capture_frame = capture_interval_s > 0 and current_time >= start_time + capture_i * capture_interval_s * 1000
            frame_img = self.draw(render, capture_frame)
            if frame_img:
                pygame.image.save(frame_img, "capture/img_" + str(capture_i) + ".png")
                capture_i += 1
            if not finished and not render.in_progress():
                finished = True
                print("Render time:", (current_time-start_time)/1000, " seconds" )

        self.__render_image = render.copy()
        pygame.image.save(self.__render_image, "rt_1.png")

    # draw scene
    def draw(self, render = None, capture = False):

        # draw background
        frame_img = None
        progress = 0
        if render and capture:
            frame_img = render.copy()
            self.__surface.blit(frame_img, (0,0))
        elif render:
            progress = render.blit(self.__surface, (0, 0))
        else:
            self.__surface.fill(BLACK)

        # draw red line which indicates the progress of the rendering
        if render and render.in_progress(): 
            progress_len = int(self.__surface.get_width() * progress)
            pygame.draw.line(self.__surface, BLACK, (0, 0), (progress_len, 0), 1) 
            pygame.draw.line(self.__surface, RED, (0, 2), (progress_len, 2), 3) 
            pygame.draw.line(self.__surface, BLACK, (0, 4), (progress_len, 4), 1) 

        # update display
        pygame.display.flip()

        return frame_img

    # init pygame display surface
    def __init_surface(self, size):
        self.__surface = pygame.display.set_mode(size, pygame.RESIZABLE)

    # handle events in a loop
    def __handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__run = False
            elif event.type == pygame.VIDEORESIZE:
                self.__init_surface((event.w, event.h))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.__run = False


###################################################################################################
# render thread
class Rendering:
    def __init__(self, world, cam):
        self.__world = world
        self.__cam = cam

    def start(self, size, no_sample, update_rate):
        self.__size = size
        self.__cam.aspect = self.__size[0]/self.__size[1]
        self.__no_samples = no_sample
        self.__update_rate = update_rate
        self.__pixel_count = 0
        self.__progress = 0
        self.__image = pygame.Surface(self.__size)
        self._stopped = False
        self.__thread = threading.Thread(target = self.run)
        self.__thread_lock = threading.Lock()
        self.__thread.start()

    # check if thread is "running"
    def in_progress(self):
        return self.__thread.is_alive()

    # wait for thread to end
    def wait(self, timeout = None):
        self.__thread.join(timeout)

    # terminate
    def stop(self):
        self.__thread_lock.acquire() 
        self._stopped = True
        self.__thread_lock.release() 
        self.__thread.join()

    # blit to surface
    def blit(self, surface, pos):
        self.__thread_lock.acquire()
        surface.blit(self.__image, pos) 
        progress = self.__progress
        self.__thread_lock.release() 
        return progress

    # copy render surface
    def copy(self):
        self.__thread_lock.acquire() 
        image = self.__image.copy()
        self.__thread_lock.release() 
        return image 

    def run(self):
        no_samples = self.__no_samples
        count = 0
        colarr = [0] * (self.__size[0] * self.__size[1] * 3)
        
        for x in range(self.__size[0]):
            for y in range(self.__size[1]):
                col = vec3()
                for s in range(no_samples):
                    u, v = (x + rand01()) / self.__size[0], (y + rand01()) / self.__size[1]
                    r = self.__cam.get_ray(u, v)
                    col += Rendering.rToColor(r, self.__world, 0)
                
                arri = y * self.__size[0] * 3 + x * 3
                colarr[arri+0] += col[0] 
                colarr[arri+1] += col[1]
                colarr[arri+2] += col[2] 
                col = vec3(colarr[arri+0], colarr[arri+1], colarr[arri+2])    

                self.__thread_lock.acquire()
                surfaceSetXYWHf(self.__image, x, y, 1, 1, col / no_samples)
                self.__thread_lock.release()
                
                count += 1
                self.__progress = count / (no_samples * self.__size[0] * self.__size[1])
                
                if self._stopped:
                    break

    max_dist = 1e20
    @staticmethod
    def rToColor(r, world, depth):
        rec = world.hit(r, 0.001, Rendering.max_dist) 
        if rec:
            if depth >= 50: #并不准确的限制，为了减少渲染时间
                return vec3(0, 0, 0) # TODO !!!
            sc_at = rec.material.scatter(r, rec)
            if not sc_at:
                return vec3(0, 0, 0)
            return multiply_components(sc_at[1], Rendering.rToColor(sc_at[0], world, depth+1)) #递归

        unit_direction = r.direction.normalize()
        t = 0.5 * (unit_direction.y + 1)
        return (1-t)*vec3(1, 1, 1) + t*vec3(0.5, 0.7, 1)#简单模拟天空的颜色

# color conversions
def fToColor(c):
    # gc = c; 
    gc = (math.sqrt(c[0]), math.sqrt(c[1]), math.sqrt(c[2])) # gamma 2
    return (int(gc[0]*255.5), int(gc[1]*255.5), int(gc[2]*255.5))
def surfaceSetXYWHi(s, x, y, w, h, c):
    if c[0] > 255 or c[1] > 255 or c[2] > 255:
        c = (max(0, min(255, c[0])), max(0, min(255, c[1])), max(0, min(255, c[2])))
    if w > 1 or h > 1: 
        s.fill(c, (x, s.get_height()-y-h, w, h))
    else:
        s.set_at((x, s.get_height()-y-1), c)
def surfaceSetXYWHf(s, x, y, w, h, c):
    surfaceSetXYWHi(s, x, y, w, h, fToColor(c))

def multiply_components(a, b):
    return vec3(a[0]*b[0],a[1]*b[1],a[2]*b[2])
###################################################################################################
# main
app = Application((600, 400), caption = "Shallow Path Tracing")
size = app.size 
scene = Scene(size)
world, cam = scene.create_scene()
#world, cam = scene.random_scene()
render = Rendering(world, cam)
app.run(render, 100, 1, 0)
    
    
    

    