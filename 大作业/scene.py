import pygame
import random
from hit import HitableList
from material import Lambertian, Metal, Dielectric
from object import Sphere, Cuboid
from camera import Camera

vec3 = pygame.Vector3
rand01 = random.random

class Scene:
    def __init__(self, size):
        self.size = size 
    
    def random_scene(self):
        objects = HitableList()
        objects.append(Sphere(vec3(0, -1000, 0), 1000, Lambertian(vec3(0.5, 0.5, 0.5))))

        for a in range(-5, 5):
            for b in range(-5, 5):
                choose_mat = rand01()
                center = vec3(a+0.9*rand01(), 0.2, b+0.9*rand01())
                if (center-vec3(4, 0.2, 0)).magnitude() > 0.9:
                    if choose_mat < 0.4:
                        # diffuse
                        mat = Lambertian(vec3(rand01()*rand01(), rand01()*rand01(), rand01()*rand01()))
                        objects.append(Cuboid(center, mat, 0.2, 0.2, 0.2))  
                    if choose_mat < 0.8:
                        # diffuse
                        mat = Lambertian(vec3(rand01()*rand01(), rand01()*rand01(), rand01()*rand01()))
                        objects.append(Sphere(center, 0.2, mat)) 
                    elif choose_mat < 0.87:
                        # metal
                        mat = Metal(vec3(0.5*(1+rand01()), 0.5*(1+rand01()), 0.5*(1+rand01())), 0.5*rand01())
                        objects.append(Sphere(center, 0.2, mat))
                    elif choose_mat < 0.95:
                        # metal
                        mat = Metal(vec3(0.5*(1+rand01()), 0.5*(1+rand01()), 0.5*(1+rand01())), 0.5*rand01())
                        objects.append(Cuboid(center, mat, 0.2, 0.2, 0.2))
                    else:
                        # glass
                        mat = Dielectric(1.5)
                        objects.append(Sphere(center, 0.2, mat))

        objects.append(Sphere(vec3(0, 1, 0), 1, Dielectric(1.5)))
        objects.append(Sphere(vec3(-4, 1, 0), 1, Lambertian(vec3(0.4, 0.2, 0.1))))
        objects.append(Cuboid(vec3(4, 1, 0), Metal(vec3(0.7, 0.6, 0.5), 0.0), 1.5, 1.5, 1.5))

        lookfrom = vec3(12, 2, 3)
        lookat = vec3(0, 0, 0)
        dist_to_focus = 10
        #dist_to_focus = (lookat-lookfrom).magnitude()
        aperture = 0.1
        cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, self.size[0]/self.size[1], aperture, dist_to_focus)
        return objects, cam

    def create_scene(self):
        objects = HitableList()
        objects += [
            #Sphere(vec3(0, 0, -1), 0.5,      Lambertian(vec3(0.1, 0.2, 0.5))),
            Sphere(vec3(0, -100.5, -1), 100, Lambertian(vec3(0.8, 0.8, 0))),
            Sphere(vec3(2, 0, -1), 0.5,      Metal(vec3(0.2, 0.6, 0.8), 0.2)),
            Sphere(vec3(-2, 0, -1), 0.5,     Dielectric(1.5)),
            Sphere(vec3(-2, 0, -1), -0.45,   Dielectric(1.5)),
            Cuboid(vec3(-2, 3, -1), Metal(vec3(0.1, 0.2, 0.5)), 0.7, 0.7, 0.7),
            Cuboid(vec3(0, 0, -1), Metal(vec3(0.8, 0.6, 0.2)), 0.5, 0.5, 0.5),
            Cuboid(vec3(0, 2, -1), Dielectric(1.5), 0.5, 1, 1),
            Sphere(vec3(-4, 1, 0), 1, Lambertian(vec3(0.4, 0.2, 0.1))),
            Sphere(vec3(4, 1, 1), 1, Metal(vec3(0.7, 0.6, 0.5), 0.0))
        ]
        lookfrom = vec3(10, 10, 20)
        lookat = vec3(0, 0, -1)
        dist_to_focus = (lookat-lookfrom).magnitude()
        aperture = 0.5
        cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, self.size[0]/self.size[1], aperture, dist_to_focus)
        return objects,cam 