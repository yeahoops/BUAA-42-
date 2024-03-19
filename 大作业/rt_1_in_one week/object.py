import numpy as np
from hit import HitRecord, Hitable
import pygame

vec3 = pygame.Vector3

# sphere hitable object
class Sphere(Hitable):
    def __init__(self, center, radius, material):
        super().__init__()
        self.__center = center
        self.__radius = radius
        self.__material = material

    def hit(self, r, tmin, tmax):
        oc = r.origin - self.__center
        a = r.direction.dot(r.direction)
        b = 2 * oc.dot(r.direction)
        c = oc.dot(oc) - self.__radius*self.__radius
        discriminant = b*b - 4*a*c
        if discriminant > 0:
            temp = (-b - np.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - self.__center) / self.__radius, self.__material )
            temp = (-b + np.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - self.__center) / self.__radius, self.__material )
        
        return None

class Cuboid(Hitable):
    def __init__(self, center, material, width, height, length):
        super().__init__()
        self.center = center
        self.width = width
        self.height = height
        self.length = length
        self.material = material
        self.lb = self.center - vec3(width/2, height/2, length/2)
        self.rt = self.center + vec3(width/2, height/2, length/2)
        self.lb_local_basis = self.lb
        self.rt_local_basis = self.rt

        # basis vectors
        self.ax_w = vec3(1.,0.,0.)
        self.ax_h = vec3(0.,1.,0.)
        self.ax_l = vec3(0.,0.,1.)

        self.inverse_basis_matrix = np.array([[self.ax_w[0],       self.ax_h[0],         self.ax_l[0]],
                                              [self.ax_w[1],       self.ax_h[1],         self.ax_l[1]],
                                              [self.ax_w[2],       self.ax_h[2],         self.ax_l[2]]])

        self.basis_matrix = self.inverse_basis_matrix.T
    def hit(self, r, tmin, tmax):
        UPWARDS = 1
        UPDOWN = -1         
        FARAWAY = 1.0e39
        
        O_local_basis = np.dot(r.origin, self.basis_matrix)
        D_local_basis = np.dot(r.direction, self.basis_matrix)

        dirfrac = 1.0 / D_local_basis
  

        t1 = (self.lb_local_basis[0] - O_local_basis[0])*dirfrac[0]
        t2 = (self.rt_local_basis[0] - O_local_basis[0])*dirfrac[0]
        t3 = (self.lb_local_basis[1] - O_local_basis[1])*dirfrac[1]
        t4 = (self.rt_local_basis[1] - O_local_basis[1])*dirfrac[1]
        t5 = (self.lb_local_basis[2] - O_local_basis[2])*dirfrac[2]
        t6 = (self.rt_local_basis[2] - O_local_basis[2])*dirfrac[2]

        t_min = np.maximum(np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)), np.minimum(t5, t6))
        t_max = np.minimum(np.minimum(np.maximum(t1, t2), np.maximum(t3, t4)), np.maximum(t5, t6))

        mask1 = (t_max < 0) | (t_min > t_max)
        mask2 = t_min < 0
        hit_info = np.select([mask1,mask2,True] , [FARAWAY , [t_max] ,  [t_min]])

        if hit_info is not None and tmin < hit_info[0] < tmax:
            return HitRecord(hit_info[0], r.point_at_parameter(hit_info[0]) , self.get_Normal(r.point_at_parameter(hit_info[0])), self.material)
        else:
            return None

   
    def get_Normal(self, point):
        P = np.dot(point - self.center, self.basis_matrix)
        absP = np.abs(P)
        Pmax = np.max([absP[0], absP[1], absP[2]])
        P[0] = np.where(Pmax == absP[0], np.sign(P[0]),  0.)
        P[1] = np.where(Pmax == absP[1], np.sign(P[1]),  0.)
        P[2] = np.where(Pmax == absP[2], np.sign(P[2]),  0.)

        return np.dot(P, self.inverse_basis_matrix)


