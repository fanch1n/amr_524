import numpy as np
#import mesh, solver

class Field:
    def __init__(self, nx, ny):
        self.nx = nx #number of pixels on the x axis
        self.ny = ny #number of pixels on the y axis
        self.mesh = np.ones((nx, ny))
        self.values = np.ones((nx, ny))
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1

    '''initialize the field with a regular mesh nx*ny, file is a file that contains (x, y, f(x, y)) data. Do a linear interpolation '''
    def initialize_values(self, file_path):
        #read user input
        with open(file_path) as textFile:
            lines = np.array([line.split() for line in textFile])
        self.x_min = min(lines[:, 0])
        self.x_max = max(lines[:, 0])
        self.y_min = min(lines[:, 1])
        self.y_max = max(lines[:, 1])
        dx = (self.x_max - self.x_min)/self.nx
        dy = (self.y_max - self.y_min)/self.ny
        return

    def get_nx(self):
        return self.nx

    def get_ny(self):
        return self.ny

    def get_mesh(self):
        return self.mesh

    def evolve(self, ode, dt, t):
        ode.evolve(self.mesh, dt, t)

my_field = Field(4, 4)
my_field.initialize_values("lib/field.xyz")
print(my_field.x_max)
