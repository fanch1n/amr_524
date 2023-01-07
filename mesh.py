#==============================================================================
# pkgs and global variables
# need numpy, matplotlib to run the program
#==============================================================================
# Open cv library
#import cv2

# matplotlib for displaying the images 
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import random
import math
import numpy as np
import copy
import time


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
    
    def get_points(self, domain):
        return domain[self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]

    def get_grad(self, domain):
        pixels = self.get_points(domain)
        if (np.size(pixels) <= 1):
            return 0
        #print(np.size(pixels))
        x_grad = np.gradient(pixels)[0]
        y_grad = np.gradient(pixels)[1]
        grad_phi = np.sqrt(x_grad**2 + y_grad**2)
        
        #return grad_phi
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
    
    def graph_tree(self, save=None):
        fig = plt.figure(figsize=(10, 10))
        #plt.title("Mesh")
        c = find_children(self.root)
        print("Number of mini-domains: %d" %len(c))
        plt.imshow(self.domain, cmap='OrRd')
        for n in c:
            # test computation can be down here.
            #print(n.x0, n.y0, n.height, n.width)
            #print(self.domain[n.x0, n.y0])
            #self.domain[n.x0, n.y0] += 1.
            #pass
            plt.gcf().gca().add_patch(patches.Rectangle((n.y0, n.x0), n.height, n.width, fill=False))
        plt.gcf().gca().set_xlim(0, self.domain.shape[1])
        plt.gcf().gca().set_ylim(self.domain.shape[0], 0)
        plt.axis('equal')
        plt.colorbar()
        if save:
            plt.savefig(save+'.png')
            plt.close()
        #plt.show()

        return

#==============================================================================
# functions to implement adaptive mesh using the Classes
#==============================================================================
 
def recursive_subdivide(node, k, minCellSize, domain):
 
    #if node.get_grad(domain).any() <=k:
    if node.get_grad(domain) <=k:
       return 0
    w_1 = int(math.floor(node.width/2)) # round down
    w_2 = int(math.ceil(node.width/2)) # round up
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))

    if w_1 < minCellSize or h_1 < minCellSize: # if w_1 or h_1 <= 1/2, skip subdivision
        return 0
    
    x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    recursive_subdivide(x1, k, minCellSize, domain)

    x2 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm left
    recursive_subdivide(x2, k, minCellSize, domain)

    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    recursive_subdivide(x3, k, minCellSize, domain)

    x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
    t = recursive_subdivide(x4, k, minCellSize, domain)
    
    node.children = [x1, x2, x3, x4]
    
    #if (not t):
    #    recursive_subdivide(Node(node.x0, node.y0, 2*w_2, 2*h_2), 0, minCellSize, domain)
    
    return 1
   

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

#==============================================================================
# functions for computation
#==============================================================================
def laplacian(M, idx, idy, w, h, NX=256, NY=256, bc = 1): # bc = 1 corresponds to periodic
    # define nearest neighbours
    xp = idx + w
    xm = idx - w
    yp = idy + h
    ym = idy - h
    
    # modify coordinates for PBC
    if (bc == 1):
        if xm < 0:
            xm = NX - w
        if xp > NX - 1:
            xp = 0
        if ym < 0:
            ym = NY - h
        if yp > NY - 1:
            yp = 0
            
        #print(idx, idy, w, h, xm, xp, ym, yp)
        #print(M[idx, idy])
        #print(M[xm, idy])
        #print(M[xp, idy])
        lap_value = (M[xm, idy] - 2. * M[idx, idy] + M[xp, idy]) / w**2 \
            + (M[idx, ym] - 2. * M[idx, idy] + M[idx, yp]) / h**2
        #print("Laplacian = ", lap_value)
        return lap_value

def evolve_diffusion(u, D = 1, dt = 0.01):
    field_old = u.domain
    field_new = np.copy(field_old)
    nodes = find_children(u.root)
    for n in nodes:
        #print(n.x0, n.y0, n.width, n.height)
        field_new[n.x0, n.y0] += D * dt * laplacian(field_old, n.x0, n.y0, n.width, n.height)
        field_new[n.x0 + n.width-1, n.y0] += D * dt * laplacian(field_old, n.x0 + n.width-1, n.y0, n.width, n.height)
        field_new[n.x0, n.y0 + n.height-1] += D * dt * laplacian(field_old, n.x0, n.y0 + n.height-1, n.width, n.height)
        field_new[n.x0 + n.width-1, n.y0 + n.height-1] += D * dt * laplacian(field_old, n.x0 + n.width-1, n.y0 + n.height-1, n.width, n.height)

    return field_new

def evolve_CH(u, D = 5, dt = 0.01):
    field_old = u.domain
    L = field_old.shape[0]
    dfdc = np.zeros_like(field_old)
    field_new = np.copy(field_old)
    nodes = find_children(u.root)

    for i in range(0,L):
        for j in range(0,L):
            dfdc[i,j] = -field_old[i,j]+(field_old[i,j])**3 - laplacian(field_old, i, j, 1, 1)
    
    for n in nodes:
        field_new[n.x0, n.y0] += D * dt * laplacian(dfdc, n.x0, n.y0, n.width, n.height)
        field_new[n.x0 + n.width-1, n.y0] += D * dt * laplacian(dfdc, n.x0 + n.width-1, n.y0, n.width, n.height)
        field_new[n.x0, n.y0 + n.height-1] += D * dt * laplacian(dfdc, n.x0, n.y0 + n.height-1, n.width, n.height)
        field_new[n.x0 + n.width-1, n.y0 + n.height-1] += D * dt * laplacian(dfdc, n.x0 + n.width-1, n.y0 + n.height-1, n.width, n.height)

    return field_new

def compare(field, field_ref):
    L_half = int(field.shape[0]/2)
    x = np.arange(L_half)
    T = []
    T_ref = []
    for i in range(L_half):
        T.append(field[L_half, i])
        T_ref.append(field[L_half, i])
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, T, 'k', linewidth = 2, label='Uniform mesh, dx = 1')
    plt.plot(x, T_ref, 'r--', linewidth = 2, label='Adaptive mesh')
    
    plt.legend()
    plt.xlim([120, 255])
    #plt.show()
    plt.savefig("diffusion_comp.png")
    
# main function to test
def main():
    
    # Define/load constant parameters
    EPS = 1e-3
    CELL_MIN = 2
    
    # Load a sample 2D matrix
    #data_init = np.load("circle.npy")
    data_init = 0.1*(np.random.rand(256, 256)-0.5) - 0.4
    Lx, Ly = data_init.shape[0], data_init.shape[1] 
    print("Size of the system: ", (Lx, Ly))

    #write_field(domain, 0)
    #domain = cv2.imread('final.png')
    
    # Initial data and its deep copy
    data = QTree(EPS, CELL_MIN, data_init)
    data.subdivide()
    data.graph_tree('step0')
    print("Total field value: ", np.sum(data.domain))

    t0 = time.time()
    for step in range(1, 200001):
        #data.domain = np.copy(evolve_diffusion(data))
        data.domain = np.copy(evolve_CH(data))
        if step % 1000 == 0:
            data = QTree(EPS, CELL_MIN, data.domain)
            data.subdivide()
            data.graph_tree('step%d'%step)
            
            print("Total field value: ", np.sum(data.domain))
    #data.graph_tree('step%d'%10000)
    t1 = time.time()
    #np.save("final.npy", data.domain)
    print("Program finished. Running time: {0}".format(t1-t0))
    
    #data_ref = np.load("ref.npy")
    #data_final = np.load("final.npy")
    #compare(data_final, data_ref)
    
if __name__ == '__main__':
    main()