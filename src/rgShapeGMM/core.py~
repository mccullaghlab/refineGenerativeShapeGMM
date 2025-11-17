import numpy as np
from shapeGMMTorch import ShapeGMM
# load library to calculate MM energies
from . import mm
# load library to save 3Nx3N generated GMMs
from . import gmm3n

def spectral_inverse(M):
    e, v = np.linalg.eigh(M)
    e[6:] = 1/e[6:]
    e[:6] = 0.0
    return v @ np.diag(e) @ v.T

def refine_gmm(shapeGMMobj, parm_file_name, atom_selection):
    # extract meta data
    n_components = shapeGMMobj.n_components
    n_atoms = shapeGMMobj.n_atoms
    
    # assuming atomselection is the appropriate remove_indeces
    remove_indeces = atom_selection
    # declare MM object and remove indeces
    mm_obj = mm.mm(parm_file_name)
    mm_obj.remove_indeces(remove_indeces)

    # align each cluster and compute covars
    covars = np.empty((n_components,n_atoms*3,n_atoms*3))
    centers = np.empty((n_components,n_atoms,3))
    kbT = 0.598 # in kcal/mol at 300K
    weights = shapeGMMobj.weights_
    for cid in range(n_components):
        # Determine the structure that mininimizes the energy starting with each mean
        mm_obj.energy_minimize(shapeGMMobj.means_[cid])
        centers[cid] = mm_obj.min_pos
        # Compute the covariance from the Hessian
        covars[cid] = kbT* spectral_inverse(mm_obj.total_hessian)

    # create the gmm3N object with precomputed values
    gmm3n_obj = gmm3n.gmm3N(centers,covars,weights)
    
    return gmm3n_obj
