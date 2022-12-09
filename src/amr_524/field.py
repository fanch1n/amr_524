import numpy as np
import matplotlib.pyplot as plt


class Field:
    def __init__(self, nx, ny):
        self.nx = nx  # number of pixels on the x axis
        self.ny = ny  # number of pixels on the y axis
        # field values, i.e., f(x, y)
        self.values = np.ones((nx, ny))
        # (x, y) coordinates of the mesh grid points
        self.x_coord = np.linspace(0, 1, self.nx)
        self.y_coord = np.linspace(0, 1, self.ny)

    """initialize the field with a regular mesh nx*ny,
    file_path is a file path that contains (x, y, f(x, y)) data.
    Do a voronoi value assignment, that is, the value assigned to a grid point p
    is the value at the closest (x, y) point in the external file """

    def initialize_values(self, file_path, strategy="voronoi"):
        with open(file_path) as textFile:
            file_data = np.array([line.split() for line in textFile])
        file_data = np.asarray(file_data, dtype=float)
        x_min = min(file_data[:, 0])
        x_max = max(file_data[:, 0])
        y_min = min(file_data[:, 1])
        y_max = max(file_data[:, 1])
        self.x_coord = np.linspace(x_min, x_max, self.nx)
        self.y_coord = np.linspace(y_min, y_max, self.ny)
        for i in range(self.nx):
            x = self.x_coord[i]
            for j in range(self.ny):
                y = self.y_coord[j]
                if strategy == "voronoi":
                    smallest_dist_rows = [
                        file_data[0]
                    ]  # rows that give smallest distances
                    smallest_dist = (
                        x_max + y_max
                    )  # smallest distance from data point to current pont
                    for row in file_data:  
                    # this is slower than cell list, 
                    # but it's just an initialization so the cost is not that big
                        (x_file, y_file) = (row[0], row[1])
                        dist = np.sqrt((x - x_file) ** 2 + (y - y_file) ** 2)
                        if dist < smallest_dist:
                            smallest_dist_rows = [row]
                            smallest_dist = dist
                        elif dist == smallest_dist:
                            smallest_dist_rows.append(row)
                    val = 0
                    for row in smallest_dist_rows:
                        val += row[2]
                    self.values[i, j] = val / len(smallest_dist_rows)
                elif strategy == "inverse_dist":
                    weight_sum = 0
                    for row in file_data:
                        (x_file, y_file) = (row[0], row[1])
                        dist = np.sqrt((x - x_file) ** 2 + (y - y_file) ** 2)
                        if dist == 0:
                            self.values[i, j] = row[2]
                            break  # break the innermost loop only
                        else:
                            weight = 1 / dist
                            weight_sum += weight
                            self.values[i, j] += weight * row[2]
                    if dist != 0:
                        self.values[i, j] /= weight_sum
        return

    """raw gradient with the finest mesh. No stencil, just the regular thing"""

    def gradient(self, boundary="absorbing"):
        result = np.zeros((self.nx, self.ny, 2))
        dx = self.x_coord[1] - self.x_coord[0]
        dy = self.y_coord[1] - self.y_coord[0]
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x_lower = self.values[i - 1, j]
                x_higher = self.values[i + 1, j]
                result[i, j, 0] = (x_higher - x_lower) / (2 * dx)
                y_lower = self.values[i, j - 1]
                y_higher = self.values[i, j + 1]
                result[i, j, 1] = (y_higher - y_lower) / (2 * dy)
        for i in range(1, self.nx - 1):
            result[i, 0, 0] = (self.values[i + 1, 0] - self.values[i - 1, 0]) / (2 * dx)
            result[i, self.ny - 1, 0] = (
                self.values[i + 1, self.ny - 1] - self.values[i - 1, self.ny - 1]
            ) / (2 * dx)
        for j in range(1, self.ny - 1):
            result[0, j, 1] = (self.values[0, j + 1] - self.values[0, j - 1]) / (2 * dy)
            result[self.nx - 1, j, 1] = (
                self.values[self.nx - 1, j + 1] - self.values[self.nx - 1, j - 1]
            ) / (2 * dy)
        if boundary == "absorbing":
            for i in range(self.nx):
                result[i, 0, 1] = (self.values[i, 1] - 0) / (2 * dy)
                result[i, self.ny - 1, 1] = (0 - self.values[i, self.ny - 2]) / (2 * dy)
            for j in range(self.ny):
                result[0, j, 0] = (self.values[1, j] - 0) / (2 * dx)
                result[self.nx - 1, j, 0] = (0 - self.values[self.nx - 2, j]) / (2 * dx)
        elif boundary == "periodic":
            for i in range(self.nx):
                result[i, 0, 1] = (self.values[i, 1] - self.values[i, self.ny - 1]) / (
                    2 * dy
                )
                result[i, self.ny - 1, 1] = (
                    self.values[i, 0] - self.values[i, self.ny - 2]
                ) / (2 * dy)
            for j in range(self.ny):
                result[0, j, 0] = (self.values[1, j] - self.values[self.nx - 1, j]) / (
                    2 * dx
                )
                result[self.nx - 1, j, 0] = (
                    self.values[0, j] - self.values[self.nx - 2, j]
                ) / (2 * dx)
        return result

    def visualize(self, ncontours=10):
        result = plt.figure(1)
        plt.contourf(
            self.x_coord, self.y_coord, self.values, ncontours, cmap=plt.cm.rainbow
        )
        plt.colorbar()
        plt.axis("equal")
        plt.title("Field values")
        return result

    def visualize_gradient(self, boundary="absorbing"):
        grad = self.gradient(boundary)
        u = grad[:, :, 0]
        v = grad[:, :, 1]
        result = plt.figure(2, figsize=(5, 5))
        plt.quiver(self.x_coord, self.y_coord, v, u, color="g", clip_on=False)
        plt.title("Field gradient")
        return result

    def evolve(self, pde, dt, time_steps):
        pde.evolve(self.mesh, dt, time_steps)
        return

    def __str__(self):
        s = ""
        for i in range(self.nx):
            for j in range(self.ny):
                s += (
                    str(self.x_coord[i])
                    + " "
                    + str(self.y_coord[j])
                    + " "
                    + str(self.values[i, j])
                    + "\n"
                )
        return s
