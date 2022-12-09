from amr_524.field import Field
import amr_524.mesh as mesh
import amr_524.evolvQtree as evolvQtree
import numpy as np


def test_field_initialize_values():
    my_field = Field(10, 10)
    my_field.initialize_values("lib/field.xyz", "voronoi")
    assert my_field.values[-1, -2] == 2.0
    return


def test_field_mesh():
    my_field = Field(250, 250)
    my_field.initialize_values("lib/field.xyz", "voronoi")
    domain = my_field.values
    EPS = 1e-8
    CELL_MIN = 3
    bondx = "OPEN"
    test_domain = mesh.QTree(EPS, CELL_MIN, domain, bondx)
    test_domain.subdivide()  # recursively generates quad tree
    assert test_domain.graph_tree("lib/test_field_mesh_open") == 571
    bondx = "PBC"
    test_domain = mesh.QTree(EPS, CELL_MIN, domain, bondx)
    test_domain.subdivide()  # recursively generates quad tree
    assert test_domain.graph_tree("lib/test_field_mesh_pbc") == 910
    return


def test_circle_mesh():
    # Define/load constant parameters
    EPS = 1e-8
    CELL_MIN = 2
    bondx = "PBC"
    # Load a sample 2D matrix
    domain = np.load("lib/circle.npy")
    print("Size of the system: ", (domain.shape[0], domain.shape[1]))
    test_domain = mesh.QTree(
        EPS, CELL_MIN, domain, bondx
    )  # contrast threshold, min cell size,domaing
    test_domain.subdivide()  # recursively generates quad tree
    assert test_domain.graph_tree("lib/test_circle_mesh") == 904
    return


def test_mass_conservation():
    EPS = 1e-6
    CELL_MIN = 4
    domain = np.load("lib/circle.npy")
    test_domain = mesh.QTree(EPS, CELL_MIN, domain, bondx="PBC")
    mean_field_initial = np.mean(test_domain.domain.flatten())
    evolvQtree.evolve_n_steps(test_domain, evolvQtree.evolve_diffusion, 1000)
    mean_field_final = np.mean(test_domain.domain.flatten())
    diff = abs(mean_field_final - mean_field_initial)
    assert diff / mean_field_initial < 0.01
