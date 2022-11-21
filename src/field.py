import numpy as np
import mesh, solver

class Field:
    def __init__(self, nx, ny, mesh, values):
        self.nx = nx #number of pixels on the x axis
        self.ny = ny #number of pixels on the y axis
        self.mesh = np.ones(nx, ny)
        self.values = np.ones(nx, ny)

    def initialize(nx, ny, file):
        #read user input
        '''initialize the field with a regular mesh nx*ny, file is a file that contains (x, y, z) data. Do a linear interpolation '''
        regular_mesh = np.ones(nx, ny)
        return

    def get_nx(self):
        return self.nx

    def get_ny(self):
        return self.ny

    def get_mesh():
        return mesh

    def evolve(self, ode, dt, t):
        ode(self.mesh, dt, t)