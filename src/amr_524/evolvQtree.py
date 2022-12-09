from amr_524.mesh import QTree
import numpy as np


def evolve_diffusion(qtree, dx=1, dy=1, dt=0.01):  # evolve one step
    nodes = qtree.root.find_children()
    old_field = qtree.domain  # old field
    bondx = qtree.bondx
    CELL_MIN = qtree.minCellSize
    mesh_new = np.copy(old_field)  # new field
    Lx, Ly = old_field.shape[0], old_field.shape[1]

    def laplacian(prev, curr, nxt, cellmin):
        return (nxt - 2 * curr + prev) / cellmin

    node_values = dict()
    for node in nodes:
        # calculate gradient only for points in qtree
        corners = node.get_corners()
        for corner in corners:
            x, y = corner[0], corner[1]
            if (x, y) not in node_values:
                lap_x, lap_y = 0, 0
                if bondx == "OPEN":
                    if (
                        x > CELL_MIN - 1
                        and x < Lx - CELL_MIN
                        and y > CELL_MIN - 1
                        and y < Ly - CELL_MIN
                    ):
                        lap_x = laplacian(
                            old_field[x - CELL_MIN, y],
                            old_field[x, y],
                            old_field[x + CELL_MIN, y],
                            CELL_MIN,
                        )
                        lap_y = laplacian(
                            old_field[x, y - CELL_MIN],
                            old_field[x, y],
                            old_field[x, y + CELL_MIN],
                            CELL_MIN,
                        )
                elif bondx == "PBC":
                    lap_x = laplacian(
                        old_field[(x - CELL_MIN) % Lx, y],
                        old_field[x, y],
                        old_field[(x + CELL_MIN) % Lx, y],
                        CELL_MIN,
                    )
                    lap_y = laplacian(
                        old_field[x, (y - CELL_MIN) % Ly],
                        old_field[x, y],
                        old_field[x, (y + CELL_MIN) % Ly],
                        CELL_MIN,
                    )
                else:
                    raise Exception("missing or incorrect boundary condition")
                mesh_new[x, y] += dt * (lap_x / dx + lap_y / dy)
                node_values[(x, y)] = mesh_new[x, y]
            node.get_corners_points(mesh_new, corners.index(corner)).fill(
                node_values[(x, y)]
            )
    return mesh_new


def evolve_n_steps(qtree, pde, steps, plot_dir="lib/diffusion", **pde_kwargs):
    EPS = qtree.threshold
    CELL_MIN = qtree.minCellSize
    bondx = qtree.bondx
    qtree.subdivide()
    qtree.graph_tree(plot_dir + "step0")
    new_mesh = qtree.domain
    for step in range(steps):
        new_mesh = pde(qtree, **pde_kwargs)
        if step < 20 or step % 500 == 0:
            print(step, np.mean(new_mesh.flatten()))
            qtree = QTree(EPS, CELL_MIN, new_mesh, bondx)
            qtree.subdivide()
            if step < 20 or step % 500 == 0:
                qtree.graph_tree(plot_dir + "/step%d" % step)
        qtree.domain = np.copy(new_mesh)
    return None


if __name__ == "__main__":
    domain = np.load("lib/circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))
    test_domain = QTree(domain, EPS=1e-6, CELL_MIN=4, bondx="PBC")
    evolve_n_steps(test_domain, evolve_diffusion, 10000)
