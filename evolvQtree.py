from mesh import QTree, find_children
import copy 
import numpy as np
import matplotlib.pyplot as plt

def evolve(qtree, bondx='OPEN'):
    nodes = find_children(qtree.root)
    mesh_new = np.copy(qtree.domain)
    Lx, Ly = qtree.domain.shape[0], qtree.domain.shape[1]
    laplacian = lambda prev, curr, nxt: nxt - 2 * curr + prev
    dx, dy, dt = 1, 1, 0.01 #FIXME 
    for node in nodes:
        # calculate gradient only for points in qtree
        x, y = node.x0, node.y0
        if bondx == 'OPEN':
            if x > 0 and x < Lx - 1 and y > 0 and y < Ly - 1:
                lap_x = laplacian(qtree.domain[x-1, y], qtree.domain[x, y], qtree.domain[x+1, y])
                lap_y = laplacian(qtree.domain[x, y-1], qtree.domain[x, y], qtree.domain[x, y+1])
                mesh_new[x, y] += dt * (lap_x/dx + lap_y/dy) 
        elif bondx == 'PBC':  
            lap_x = laplacian(qtree.domain[(x-1)%Lx, y],\
                              qtree.domain[x, y], qtree.domain[(x+1)%Lx, y])
            lap_y = laplacian(qtree.domain[x, (y-1)%Ly],\
                              qtree.domain[x, y], qtree.domain[x, (y+1)%Ly])
            mesh_new[x, y] += dt * (lap_x/dx + lap_y/dy) 
        else:
            raise Exception("missing specification of boundary condition")
    return mesh_new

if __name__ == '__main__':
    # Define/load constant parameters
    EPS = 1e-8
    CELL_MIN = 2
    
    # Load a sample 2D matrix
    domain = np.load("circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))

    test_domain = QTree(EPS, CELL_MIN, domain)  #contrast threshold, min cell size,domaing
    test_domain.subdivide() # recursively generates quad tree
    test_domain.graph_tree('step0')
   
    for step in range(1, 5000):
        new_mesh = evolve(test_domain)
        if step < 10 or step % 500 == 0:
            test_domain.subdivide()
            test_domain.graph_tree('step%d'%step)
            print(step, np.max(np.abs(new_mesh - test_domain.domain).flatten()))
        test_domain.domain = np.copy(new_mesh)
