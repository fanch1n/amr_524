#==============================================================================
# pkgs and global variables
# need numpy, matplotlib to run the program
#==============================================================================
# Open cv library
import cv2

# matplotlib for displaying the images 
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import random
import math
import numpy as np


#==============================================================================
# classes 
#==============================================================================
# Generating adaptive meshes for a field 

"""
define the class of problem called "Node" and "QTree"
"""

class Node():
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
    
    #def get_points(self):
    #    return self.points
    
    def get_points(self, domain):
        return domain[self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]

    def get_grad(self, domain):
        pixels = self.get_points(domain)
        x_grad = np.gradient(pixels)[0]
        y_grad = np.gradient(pixels)[1]
        grad_phi = np.sqrt(x_grad**2 + y_grad**2)
        
        return np.mean(grad_phi)

class QTree():
    def __init__(self, stdThreshold, minCellSize, domain):
        self.threshold = stdThreshold
        self.minCellSize = minCellSize
        self.domain = domain
        self.root = Node(0, 0, domain.shape[0], domain.shape[1]) # Tree structure starts from (0,0)

    def get_points(self):
        return self.domain[self.root.x0:self.root.x0 + self.root.get_width(), self.root.y0:self.root.y0+self.root.get_height()]
    
    def subdivide(self):
        recursive_subdivide(self.root, self.threshold, self.minCellSize, self.domain)
    
    def graph_tree(self):
        fig = plt.figure(figsize=(10, 10))
        plt.title("Mesh")
        c = find_children(self.root)
        print("Number of mini-domains: %d" %len(c))
        plt.imshow(self.domain)
        for n in c:
            # test computation can be down here.
            #print(n.x0, n.y0, n.height, n.width)
            #print(self.domain[n.x0, n.y0])
            #self.domain[n.x0, n.y0] += 1.
            plt.gcf().gca().add_patch(patches.Rectangle((n.y0, n.x0), n.height, n.width, fill=False))
        plt.imshow(self.domain)
        plt.gcf().gca().set_xlim(0, self.domain.shape[1])
        plt.gcf().gca().set_ylim(self.domain.shape[0], 0)
        plt.axis('equal')
        #plt.colorbar()
        plt.show()
        return

#==============================================================================
# functions
#==============================================================================
# to implement adaptive mesh using the Classes
 
def recursive_subdivide(node, k, minCellSize, domain):

    if node.get_grad(domain)<=k:
        return
    w_1 = int(math.floor(node.width/2)) # round down
    w_2 = int(math.ceil(node.width/2)) # round up
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))

    if w_1 <= minCellSize or h_1 <= minCellSize:
        return
    x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    recursive_subdivide(x1, k, minCellSize, domain)

    x2 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm left
    recursive_subdivide(x2, k, minCellSize, domain)

    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    recursive_subdivide(x3, k, minCellSize, domain)

    x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
    recursive_subdivide(x4, k, minCellSize, domain)

    node.children = [x1, x2, x3, x4]
   

def find_children(node):
   if not node.children:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children

def write_field(field, step):
    plt.figure()
    plt.imshow(field[1:-1,1:-1])
    plt.axis('off')
    plt.show()
    #plt.colorbar()
    #plt.savefig('final{0:03d}.png'.format(step), dpi=300)
    #plt.close()

# main function to test
def main():
    # Define/load constant parameters
    EPS = 1e-8
    CELL_MIN = 2
    
    # Load a sample 2D matrix
    domain = np.load("final.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))

    #write_field(domain, 0)
    #domain = cv2.imread('final.png')

    test_domain = QTree(EPS, CELL_MIN, domain)  #contrast threshold, min cell size,domaing
    test_domain.subdivide() # recursively generates quad tree
    test_domain.graph_tree()

if __name__ == '__main__':
    main()