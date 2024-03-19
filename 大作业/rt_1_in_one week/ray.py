
# ray
class Ray:
    def __init__(self, a, b):
        self.A = a
        self.B = b
    @property 
    def origin(self): 
        return self.A
    @origin.setter
    def origin(self, o): 
        self.A = o
    @property 
    def direction(self): 
        return self.B
    @direction.setter
    def direction(self, d): 
        self.B = d
    def point_at_parameter(self, t):
        return self.A + self.B * t
