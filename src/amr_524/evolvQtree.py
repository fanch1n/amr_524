from mesh import QTree
import numpy as np

def evolve_diffusion(qtree): #evolve one step
    nodes = qtree.root.find_children()
    old_field = qtree.domain #old field
    bondx = qtree.bondx
    CELL_MIN = qtree.minCellSize
    mesh_new = np.copy(old_field) #new field
    Lx, Ly = old_field.shape[0], old_field.shape[1]
    laplacian = lambda prev, curr, nxt, cellmin: (nxt - 2 * curr + prev)/cellmin
    dx, dy, dt = 1, 1, 0.01 #FIXME 
    node_values = dict()
    for node in nodes:
        # calculate gradient only for points in qtree
        corners = node.get_corners()
        for corner in corners:
            x, y = corner[0], corner[1]
            if (x, y) not in node_values:
                lap_x, lap_y = 0, 0
                if bondx == 'OPEN':
                    if x > CELL_MIN - 1 and x < Lx - CELL_MIN and y > CELL_MIN - 1 and y < Ly - CELL_MIN:
                        lap_x = laplacian(old_field[x - CELL_MIN, y], old_field[x, y], old_field[x + CELL_MIN, y], CELL_MIN)
                        lap_y = laplacian(old_field[x, y - CELL_MIN], old_field[x, y], old_field[x, y + CELL_MIN], CELL_MIN)
                elif bondx == 'PBC':  
                    lap_x = laplacian(old_field[(x - CELL_MIN)%Lx, y],\
                                    old_field[x, y], old_field[(x + CELL_MIN)%Lx, y], CELL_MIN)
                    lap_y = laplacian(old_field[x, (y - CELL_MIN)%Ly],\
                                    old_field[x, y], old_field[x, (y + CELL_MIN)%Ly], CELL_MIN) 
                else:
                    raise Exception("missing specification of boundary condition")   
                mesh_new[x, y] += dt * (lap_x/dx + lap_y/dy) 
                node_values[(x, y)] = mesh_new[x, y]
            node.get_corners_points(mesh_new, corners.index(corner)).fill(node_values[(x, y)])
    return mesh_new

if __name__ == '__main__':
    # Define/load constant parameters
    EPS = 1e-6
    CELL_MIN = 4
    bondx = "PBC"
    # Load a sample 2D matrix
    domain = np.load("lib/circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))

    test_domain = QTree(EPS, CELL_MIN, domain, bondx)  #contrast threshold, min cell size,domaing
    test_domain.subdivide() # recursively generates quad tree
    test_domain.graph_tree('step0')
    nodes = test_domain.root.find_children()
    new_mesh = test_domain.domain 
   
    for step in range(1, 10000):
        new_mesh = evolve_diffusion(test_domain)
        if step < 20 or step % 500 == 0:  
            print(step, np.mean(new_mesh.flatten()))
            test_domain = QTree(EPS, CELL_MIN, new_mesh, bondx)
            test_domain.subdivide()
            if step < 20 or step % 500 == 0:
                test_domain.graph_tree('lib/diffusion/step%d'%step)
        test_domain.domain = np.copy(new_mesh)
