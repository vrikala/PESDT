
from statistics import multimode
import numpy as np
from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.edge2d.mesh_geometry import Edge2DMesh
from cherab.edge2d.edge2d_plasma import Edge2DSimulation


def load_edge2d_from_PESDT(PESDT, convert_denel_to_m3 = True):

    ########################################################################
    # Start by loading in all the data from B Lowmanowski's PESDT object #

    num_cells = len(PESDT.cells)

    rv = np.zeros((num_cells, 4))
    zv = np.zeros((num_cells, 4))
    rc = np.zeros(num_cells)
    zc = np.zeros(num_cells)

    te = np.zeros(num_cells)
    ti = np.zeros(num_cells)
    ne = np.zeros(num_cells)
    ni = np.zeros(num_cells)
    n0 = np.zeros(num_cells)

    multi = 1.0
    if convert_denel_to_m3:
        multi = 1e-6

    for ith_cell, cell in enumerate(PESDT.cells):
        # extract cell centres and vertices
        rc[ith_cell] = cell.R
        zc[ith_cell] = cell.Z

        rv[ith_cell, :] = PESDT.data.rv[ith_cell, 0:4]
        zv[ith_cell, :] = PESDT.data.zv[ith_cell, 0:4]
        # Pull over plasma values to new CHERAB arrays

        te[ith_cell] = cell.te
        ti[ith_cell] = cell.te
        # Multiply by 1e-6, I think cherab wants densities in cm^-3
        
        ni[ith_cell] = cell.ni*multi
        ne[ith_cell] = cell.ne*multi
        n0[ith_cell] = cell.n0*multi

        # Set to constant value, for debugging (COMMENT OUT!)
        # Settings for masking ISP and OSP emission
        # if (cell.R >= 2.67 and cell.Z <= -1.55) or cell.R <= 2.5:
        #     ni[ith_cell] = 1.1e+16
        #     ne[ith_cell] = 1.1e+16
        #     n0[ith_cell] = 1.1e+02
        # te[ith_cell] = cell.te*0 + 10
        # ti[ith_cell] = cell.te*0 + 10
        # ni[ith_cell] = cell.ni*0 + 1e20
        # ne[ith_cell] = cell.ne*0 + 1e20
        # n0[ith_cell] = cell.n0*0 + 1e20

    #####################################################
    # Now load the simulation object with plasma values #
    rv = np.transpose(rv)
    zv = np.transpose(zv)

    edge2d_mesh = Edge2DMesh(rv, zv) #, rc, zc)

    sim = Edge2DSimulation(edge2d_mesh, [('D', 0), ('D', 1)]) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti

    # Load electron species
    #sim._electron_temperature = te
    #sim._electron_density = ne

    # Master list of species, e.g. ['D0', 'D+1', 'C0', 'C+1', ...
    num_species = 2
    #sim._species_list = ['D0', 'D+1']
    species_density = np.zeros((num_species, num_cells))
    species_density[0, :] = n0[:]  # neutral density D0
    species_density[1, :] = ni[:]  # ion density D+1
    
    sim.species_density = species_density
    #sim._species_density = species_density
    #sim._ion_temperature = ti

    # Make Mesh Interpolator function for inside/outside mesh test.
	#inside_outside_data = np.ones(edge2d_mesh._num_tris)
    #inside_outside = AxisymmetricMapper(Discrete2DMesh(edge2d_mesh.vertex_coordinates, edge2d_mesh.triangles, inside_outside_data, limit=False))
    #sim._inside_mesh = inside_outside

    return sim
