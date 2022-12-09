# ==============================================================================
# pkgs and global variables
# need numpy, matplotlib to run the program
# ==============================================================================
# Open cv library
# import cv2

# matplotlib for displaying the images
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
import math
import numpy as np


# ==============================================================================
# classes
# ==============================================================================
# Generating adaptive meshes for a field

"""
define the class of problem called "Node" and "QTree"
"""


class Node:
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self, domain):
        return domain[
            self.x0 : self.x0 + self.get_width(), self.y0 : self.y0 + self.get_height()
        ]

    def get_points_loose(self, domain, bondx="OPEN"):
        nx = domain.shape[0]
        ny = domain.shape[1]
        if bondx == "OPEN":
            smallx = max(0, self.x0 - 1)
            largex = min(nx, self.x0 + self.get_width() + 1)
            smally = max(0, self.y0 - 1)
            largey = min(ny, self.y0 + self.get_height() + 1)
            return domain[smallx:largex, smally:largey]
        if bondx == "PBC":
            domain_vcat = np.concatenate(
                (domain[-1].reshape((1, nx)), domain, domain[0].reshape((1, nx)))
            )
            domain_cat = np.hstack(
                (
                    domain_vcat[:, -1].reshape((ny + 2, 1)),
                    domain_vcat,
                    domain_vcat[:, 0].reshape((ny + 2, 1)),
                )
            )
            smallx = self.x0
            largex = self.x0 + self.get_width() + 2
            smally = self.y0
            largey = self.y0 + self.get_height() + 2
            return domain_cat[smallx:largex, smally:largey]

    def get_corners(self):
        return [
            (self.x0, self.y0),
            (self.x0 + self.width - 1, self.y0),
            (self.x0, self.y0 + self.height - 1),
            (self.x0 + self.width - 1, self.y0 + self.height - 1),
        ]

    def get_corners_points(self, domain, ind):
        if ind == 0:
            return domain[
                self.x0 : self.x0 + int(self.width / 2),
                self.y0 : self.y0 + int(self.height / 2),
            ]
        elif ind == 1:
            return domain[
                self.x0 + int(self.width / 2) : self.x0 + self.width,
                self.y0 : self.y0 + int(self.height / 2),
            ]
        elif ind == 2:
            return domain[
                self.x0 : self.x0 + int(self.width / 2),
                self.y0 + int(self.height / 2) : self.y0 + self.height,
            ]
        elif ind == 3:
            return domain[
                self.x0 + int(self.width / 2) : self.x0 + self.width,
                self.y0 + int(self.height / 2) : self.y0 + self.height,
            ]

    def get_grad(self, domain, bondx="OPEN"):
        grad = np.gradient(self.get_points_loose(domain, bondx))
        x_grad_big = grad[0]
        y_grad_big = grad[1]
        x_grad = x_grad_big[1 : 1 + self.width, 1 : 1 + self.height]
        y_grad = y_grad_big[1 : 1 + self.width, 1 : 1 + self.height]
        grad_phi = np.sqrt(x_grad**2 + y_grad**2)
        return np.mean(grad_phi)

    def recursive_subdivide(self, threshold, minCellSize, domain, bondx="OPEN"):
        if (
            self.get_grad(domain, bondx) <= threshold
        ):  # no refinement if the gradient is too small
            return
        w_1 = int(math.floor(self.width / 2))  # round down
        w_2 = int(math.ceil(self.width / 2))  # round up
        h_1 = int(math.floor(self.height / 2))
        h_2 = int(math.ceil(self.height / 2))
        if w_1 <= minCellSize or h_1 <= minCellSize:
            return
        x1 = Node(self.x0, self.y0, w_1, h_1)  # top left
        x1.recursive_subdivide(threshold, minCellSize, domain, bondx)
        x2 = Node(self.x0, self.y0 + h_1, w_1, h_2)  # btm left
        x2.recursive_subdivide(threshold, minCellSize, domain, bondx)
        x3 = Node(self.x0 + w_1, self.y0, w_2, h_1)  # top right
        x3.recursive_subdivide(threshold, minCellSize, domain, bondx)
        x4 = Node(self.x0 + w_1, self.y0 + h_1, w_2, h_2)  # btm right
        x4.recursive_subdivide(threshold, minCellSize, domain, bondx)
        self.children = [x1, x2, x3, x4]

    def find_children(self):
        if not self.children:
            return [self]
        else:
            result = []
            for child in self.children:
                result += child.find_children()
        return result


class QTree:
    def __init__(self, stdThreshold, minCellSize, domain, bondx="OPEN"):
        self.threshold = stdThreshold
        self.minCellSize = minCellSize
        self.domain = domain
        self.root = Node(
            0, 0, domain.shape[0], domain.shape[1]
        )  # Tree structure starts from (0,0)
        self.bondx = bondx

    def get_points(self):
        return self.root.get_points()

    def subdivide(self):
        self.root.recursive_subdivide(
            self.threshold, self.minCellSize, self.domain, self.bondx
        )

    """visualize the tree mesh"""

    def graph_tree(self, save=None, title="Mesh"):
        c = self.root.find_children()
        print("Number of mini-domains: %d" % len(c))
        fig = plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(self.domain, cmap="OrRd", vmin=0, vmax=1)
        for n in c:
            # test computation can be down here.
            # print(n.x0, n.y0, n.height, n.width)
            # print(self.domain[n.x0, n.y0])
            # self.domain[n.x0, n.y0] += 1.
            plt.gcf().gca().add_patch(
                patches.Rectangle((n.y0, n.x0), n.height, n.width, fill=False)
            )
        plt.gcf().gca().set_xlim(0, self.domain.shape[1])
        plt.gcf().gca().set_ylim(self.domain.shape[0], 0)
        plt.axis("equal")
        #plt.colorbar()
        if save:
            plt.savefig(save + ".png")
        else:
            plt.show()
        return len(c)


# ==============================================================================
# functions
# ==============================================================================
# to implement adaptive mesh using the Classes


def write_field(field, step):
    plt.figure()
    plt.imshow(field[1:-1, 1:-1])
    plt.axis("off")
    plt.show()
    # plt.colorbar()
    # plt.savefig('final{0:03d}.png'.format(step), dpi=300)
    # plt.close()


def test_circle_mesh():
    # Define/load constant parameters
    EPS = 1e-8
    CELL_MIN = 2
    bondx = "PBC"
    # Load a sample 2D matrix
    domain = np.load("lib/circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))
    test_domain = QTree(
        EPS, CELL_MIN, domain, bondx
    )  # contrast threshold, min cell size,domaing
    test_domain.subdivide()  # recursively generates quad tree
    assert test_domain.graph_tree("lib/test_circle_mesh") == 946
    return


if __name__ == "__main__":
    test_circle_mesh()
