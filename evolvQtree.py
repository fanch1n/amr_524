from mesh import QTree
import numpy as np
import imageio
import glob
from natsort import natsorted
import argparse

def laplacian(prev, curr, nxt, cellmin):
    return (nxt - 2 * curr + prev) / cellmin


def calculate_laplacian(field, x, y, bondx, CELL_MIN):
    Lx, Ly = field.shape[0], field.shape[1]
    lap_x, lap_y = None, None
    if bondx == "OPEN":
        if (
            x > CELL_MIN - 1
            and x < Lx - CELL_MIN
            and y > CELL_MIN - 1
            and y < Ly - CELL_MIN
        ):
            lap_x = laplacian(
                field[x - CELL_MIN, y],
                field[x, y],
                field[x + CELL_MIN, y],
                CELL_MIN,
            )
            lap_y = laplacian(
                field[x, y - CELL_MIN],
                field[x, y],
                field[x, y + CELL_MIN],
                CELL_MIN,
            )
    elif bondx == "PBC":
        lap_x = laplacian(
            field[(x - CELL_MIN) % Lx, y],
            field[x, y],
            field[(x + CELL_MIN) % Lx, y],
            CELL_MIN,
        )
        lap_y = laplacian(
            field[x, (y - CELL_MIN) % Ly],
            field[x, y],
            field[x, (y + CELL_MIN) % Ly],
            CELL_MIN,
        )
    else:
        raise Exception("missing or incorrect boundary condition")

    return lap_x, lap_y


def evolve_spinodal(qtree, dx=1, dy=1, dt=0.005):  # evolve one step
    nodes = qtree.root.find_children()
    old_field = qtree.domain  # old field
    CELL_MIN = qtree.minCellSize
    mesh_new = np.copy(old_field)  # new field
    dphi = np.copy(old_field)
    phi = np.copy(old_field)

    node_values = dict()
    for node in nodes:
        corners = node.get_corners()
        for corner in corners:
            x, y = corner[0], corner[1]
            if (x, y) not in node_values:
                lap_x, lap_y = calculate_laplacian(old_field, x, y, qtree.bondx, CELL_MIN)
                if lap_x and lap_y:
                    dphi[x, y] = (lap_x / dx + lap_y / dy)
                    phi[x, y] = -old_field[x, y] + old_field[x,y]**3 - dphi[x, y]
                    D_x, D_y = calculate_laplacian(phi, x, y, qtree.bondx, CELL_MIN)

                    mesh_new[x, y] += dt * (D_x / dx + D_y / dy)
                    node_values[(x, y)] = mesh_new[x, y]
            node.get_corners_points(mesh_new, corners.index(corner)).fill(
                node_values[(x, y)]
            )
    return mesh_new


def evolve_diffusion(qtree, dx=1, dy=1, dt=0.01):  # evolve one step
    nodes = qtree.root.find_children()
    old_field = qtree.domain  # old field
    CELL_MIN = qtree.minCellSize
    mesh_new = np.copy(old_field)  # new field
    node_values = dict()
    for node in nodes:
        # calculate gradient only for points in qtree
        corners = node.get_corners()
        for corner in corners:
            x, y = corner[0], corner[1]
            if (x, y) not in node_values:
                lap_x, lap_y = calculate_laplacian(old_field, x, y, qtree.bondx, CELL_MIN)
                if lap_x and lap_y:
                    mesh_new[x, y] += dt * (lap_x / dx + lap_y / dy)
                    node_values[(x, y)] = mesh_new[x, y]
                    node.get_corners_points(mesh_new, corners.index(corner)).fill(
                    node_values[(x, y)]
            )
    return mesh_new


def evolve_n_steps(qtree, pde, steps, plot_dir="lib/diffusion", verbose=False, **pde_kwargs):
    EPS = qtree.threshold
    CELL_MIN = qtree.minCellSize
    bondx = qtree.bondx
    qtree.subdivide()
    qtree.graph_tree(plot_dir + "step0")
    new_mesh = qtree.domain
    for step in range(steps):
        new_mesh = pde(qtree, **pde_kwargs)
        if step < 20 or step % 10 == 0:
            if verbose:
                print(step, np.mean(new_mesh.flatten()))
            qtree.subdivide()
            if step < 20 or step % 10 == 0:
                qtree.graph_tree(plot_dir + "/step%d" % step, "step %d" % step)
        qtree.domain = np.copy(new_mesh)
    return None


def create_video(plot_dir):
    images = []
    filenames = natsorted(glob.glob(plot_dir + "/step*.png"))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(plot_dir + '/movie.gif', images, duration = 0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pde', type=str, help="diffusion/spinodal/custom defined pde")
    parser.add_argument('--eps', type=float, default=1e-4, \
            help="threshold on the gradient criterion for mesh")
    parser.add_argument('--bondx', type=str, default='PBC',\
            help="boundary condition for the pde ['PBC']")
    parser.add_argument('--init-config', type=str, default='random',\
            help="path for loading the initial configuration file .npy [random]")
    parser.add_argument('--nstep', type=int, default=1000,\
            help="number of steps to run the solver for[1000]")
    parser.add_argument('--verbose', action='store_true', help="turn on verbose output [False]")
    parser.add_argument('--output', type=str, default='lib/',\
            help="path for output files")
    clargs = parser.parse_args()

    if clargs.init_config != 'random':
        domain = np.load(clargs.init_config)
    else:
        domain = np.random.uniform(-0.2, 0.2, (100, 100))
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))
    if clargs.pde == 'diffusion':
        solver = evolve_diffusion
    elif clargs.pde == 'spinodal':
        solver = evolve.spinodal
    else:
       print("custom-defined pde needed to be implemented")
       exit(1)

    test_domain = QTree(clargs.eps, 2, domain, clargs.bondx)
    evolve_n_steps(test_domain, solver, clargs.nstep, clargs.output, clargs.verbose)
    create_video(clargs.output)
