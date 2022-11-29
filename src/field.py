import numpy as np
#import mesh, solver

class Field:
    def __init__(self, nx, ny):
        self.nx = nx #number of pixels on the x axis
        self.ny = ny #number of pixels on the y axis
        self.mesh = np.ones((nx, ny)) #grid of zeros and ones. 1 means this grid point is included in the AMC coarse-grained mesh
        self.values = np.ones((nx, ny)) #field values, i.e., f(x, y), on self.mesh
        self.mesh_coord = np.ones((nx, ny, 2)) #(x, y) coordinates of the mesh grid points
        x_min = 0 
        x_max = 1
        y_min = 0
        y_max = 1

    '''initialize the field with a regular mesh nx*ny, 
    file_path is a file path that contains (x, y, f(x, y)) data. 
    Do a voronoi value assignment, that is, the value assigned to a grid point p
    is the value at the closest (x, y) point in the external file '''
    def initialize_values_voronoi(self, file_path):
        #read user input
        with open(file_path) as textFile:
            file_data = np.array([line.split() for line in textFile])
        file_data = np.asarray(file_data, dtype=float)
        self.x_min = min(file_data[:, 0])
        self.x_max = max(file_data[:, 0])
        self.y_min = min(file_data[:, 1])
        self.y_max = max(file_data[:, 1])
        dx = (self.x_max - self.x_min)/(self.nx - 1)
        dy = (self.y_max - self.y_min)/(self.ny - 1)
        for i in range(self.nx):
            x = self.x_min + i*dx
            for j in range(self.ny):
                y = self.y_min + j*dy
                self.mesh_coord[i, j, :] = [x, y]
                smallest_dist_rows = [file_data[0]]
                smallest_dist = self.x_max + self.y_max #an upper bound
                for row in file_data: #this is slower than cell list, but it's just an initialization so the cost is not that big
                    (x_file, y_file) = (row[0], row[1])
                    dist = np.sqrt((x - x_file)**2 + (y - y_file)**2)
                    if dist < smallest_dist:
                        smallest_dist_rows = [row]
                        smallest_dist = dist
                    elif dist == smallest_dist:
                        smallest_dist_rows.append(row)
                val = 0
                for row in smallest_dist_rows:
                    val += row[2]
                self.values[i, j] = val/len(smallest_dist_rows)
        return

    def evolve(self, solver_ode, dt, t):
        solver_ode.evolve(self.mesh, dt, t)
        return 

    def __str__(self):
        s = ""
        for i in range(self.nx):
            for j in range(self.ny):
                s += str(self.mesh_coord[i, j, 0]) + " " + str(self.mesh_coord[i, j, 1]) + " " + str(self.values[i, j]) + "\n"
        return s

def test_field():
    my_field = Field(5, 5)
    my_field.initialize_values_voronoi("lib/field.xyz")
    print(my_field)
    return

test_field()
