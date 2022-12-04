from mesh import QTree, find_children
import copy 
import numpy as np
import matplotlib.pyplot as plt
import pprint

def evolve(qtree, bondx='OPEN'): #evolve one step
    nodes = find_children(qtree.root)
    mesh_new = np.copy(qtree.domain) #new field
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

def evolve2(qtree, bondx='OPEN'): #evolve one step
    nodes = find_children(qtree.root)
    old_field = qtree.domain #new field
    mesh_new = np.copy(old_field) #new field
    Lx, Ly = old_field.shape[0], old_field.shape[1]
    laplacian = lambda prev, curr, nxt: nxt - 2 * curr + prev
    dx, dy, dt = 1, 1, 0.01 #FIXME 
    node_values = dict()
    for node in nodes:
        # calculate gradient only for points in qtree
        avr = 0
        for corner in node.get_corners():
            x, y = corner[0], corner[1]
            if (x, y) not in node_values:
                lap_x, lap_y = 0, 0
                if bondx == 'OPEN':
                    if x > 0 and x < Lx - 1 and y > 0 and y < Ly - 1:
                        lap_x = laplacian(old_field[x - 1, y], old_field[x, y], old_field[x + 1, y])
                        lap_y = laplacian(old_field[x, y - 1], old_field[x, y], old_field[x, y + 1])
                elif bondx == 'PBC':  
                    lap_x = laplacian(old_field[(x-1)%Lx, y],\
                                    old_field[x, y], old_field[(x+1)%Lx, y])
                    lap_y = laplacian(old_field[x, (y-1)%Ly],\
                                    old_field[x, y], old_field[x, (y+1)%Ly]) 
                else:
                    raise Exception("missing specification of boundary condition")
                mesh_new[x, y] += dt * (lap_x/dx + lap_y/dy) 
                node_values[(x, y)] = mesh_new[x, y]
            avr += node_values[(x, y)]
        avr = avr/4.0  
        node.get_points(mesh_new).fill(avr)
    return mesh_new

if __name__ == '__main__':
    # Define/load constant parameters
    EPS = 1e-3
    CELL_MIN = 2
    
    # Load a sample 2D matrix
    domain = np.load("circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))

    test_domain = QTree(EPS, CELL_MIN, domain)  #contrast threshold, min cell size,domaing
    test_domain.subdivide() # recursively generates quad tree
    test_domain.graph_tree('step0')
    nodes = find_children(test_domain.root) 
    new_mesh = test_domain.domain 
   
    for step in range(1, 5000):
        #new_mesh = evolve(test_domain)
        new_mesh = evolve2(test_domain, "OPEN")
        if step < 10 or step % 500 == 0:
            test_domain.graph_tree('step%d'%step)
            print(step, np.max(np.abs(new_mesh - test_domain.domain).flatten()))
            test_domain = QTree(EPS, CELL_MIN, new_mesh)
            test_domain.subdivide()
        test_domain.domain = np.copy(new_mesh)
