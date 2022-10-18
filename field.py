import numpy as np
import mesh, solver

class Field

<<<<<<< HEAD
if __name__ == “__main__”:
=======
if __name__ == “main”:
  pass;

def __init__(self, nx, ny, mesh, values):
  self.nx = nx
  self.ny = ny
  self.mesh = np.ones(nx, ny)
  self.values = np.ones(nx, ny)
  
def initialize(nx, ny, file):
  #read user input
  '''initialize the field with a regular mesh nx*ny, file is a file that contains (x, y, z) data. Do a linear interpolation '''
  regular_mesh = np.ones(nx, ny)
  return

def get_nx():
  

def get_mesh():
  return mesh

def evolve():
  pass 
>>>>>>> 4c04e9110cec1890b412f06327c4d7796d9138a3

