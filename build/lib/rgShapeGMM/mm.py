import numpy as np
from scipy import optimize
from MDAnalysis.lib.distances import calc_dihedrals
from scipy.interpolate import RegularGridInterpolator

def indeces_to_key(indeces, n_atoms):
    
    original = np.arange(0,n_atoms,1).astype(int)
    key = np.arange(0,n_atoms,1).astype(int)
    for index in indeces:
        key[original > index] -= 1
    return key

class atoms:
    """
    """
    def __init__(self, n_atoms, n_types, excluded_atoms_list_length):
        """
        """
        self.n_atoms = n_atoms
        self.n_types = n_types
        self.n_types2_ut = n_types*(n_types+1)//2
        self.n_types2 = n_types*n_types
        # declare other arrays yet to be filled
        self.ityp = np.empty(n_atoms,dtype=int)
        self.charges = np.empty(n_atoms)
        self.masses = np.empty(n_atoms)
        self.n_excluded_atoms = np.empty(n_atoms,dtype=int)
        self.nonbonded_parm_index = np.empty(self.n_types2,dtype=int)
        self.lj_params = np.empty((self.n_types2_ut,2),dtype=np.float64)
        self.excluded_atoms_list_length = excluded_atoms_list_length
        self.excluded_atoms_list = np.empty(excluded_atoms_list_length,dtype=int)
        self.n_pairs = n_atoms*n_atoms

    def remove_indeces(self, indeces):
        
        # index to key
        key = indeces_to_key(indeces, self.n_atoms)
        # remove indeces from mass array
        self.masses = np.delete(self.masses, indeces)
        # remove indeces from charge array
        self.charges = np.delete(self.charges, indeces)
        # remove indeces from ityp
        self.ityp = np.delete(self.ityp, indeces)
        # remove indeces from excluded_atoms_list
        new_excluded_atom_list = []
        new_n_excluded_atoms = np.copy(self.n_excluded_atoms)
        for atom1 in range(self.n_atoms):
            if atom1 not in indeces:
                offset = np.sum(self.n_excluded_atoms[:atom1])
                for e_atom in range(self.n_excluded_atoms[atom1]):
                    if self.excluded_atoms_list[offset + e_atom] in indeces:
                        new_n_excluded_atoms[atom1] -= 1
                    else:
                        new_excluded_atom_list.append(key[self.excluded_atoms_list[offset + e_atom]])
        # remove indeces from n_excluded_atoms
        self.n_excluded_atoms = np.delete(new_n_excluded_atoms, indeces)
        # remove indeces from excluded_atoms_list_length
        self.excluded_atoms_list_length = len(new_excluded_atom_list)
        self.excluded_atoms_list = np.array(new_excluded_atom_list,dtype=int)
        # remove indeces from n_atoms
        n_remove_atoms = len(indeces)
        self.n_atoms -= n_remove_atoms
        self.n_pairs = self.n_atoms*self.n_atoms

    def energy(self, pos):

        # Assert that position is of the right size
        assert pos.shape[0] == self.n_atoms, "postition array has incorrect number of atoms"
        
        e_elec = 0.0
        e_vdw  = 0.0
        for atom1 in range(self.n_atoms-1):
            for atom2 in range(atom1+1,self.n_atoms):
                # check excluded atom list
                exclude = False
                offset = np.sum(self.n_excluded_atoms[:atom1])
                for e_atom in range(self.n_excluded_atoms[atom1]):
                    if (self.excluded_atoms_list[offset + e_atom] == atom2):
                        exclude = True
                        break
                if exclude == False:
                    r = pos[atom2]-pos[atom1]
                    r_mag = np.linalg.norm(r)
                    # Calculate LJ Energy
                    r_mag6 = r_mag**6
                    lj_type = self.ityp[atom1]*self.n_types + self.ityp[atom2]
                    lj_type = self.nonbonded_parm_index[lj_type]
                    e_vdw += self.lj_params[lj_type,0]/r_mag6**2-self.lj_params[lj_type,1]/r_mag6
                    # calculate Coulomb Energy
                    e_elec += self.charges[atom1]*self.charges[atom2]/r_mag
        return e_elec, e_vdw

    def force(self, pos):
        
         # Assert that position is of the right size
        assert pos.shape[0] == self.n_atoms, "postition array has incorrect number of atoms"

        f_elec = np.zeros(pos.shape,dtype=np.float64)
        f_vdw  = np.zeros(pos.shape,dtype=np.float64)
        for atom1 in range(self.n_atoms-1):
            for atom2 in range(atom1+1,self.n_atoms):
                # check excluded atom list
                exclude = False
                offset = np.sum(self.n_excluded_atoms[:atom1])
                for e_atom in range(self.n_excluded_atoms[atom1]):
                    if (self.excluded_atoms_list[offset + e_atom] == atom2):
                        exclude = True
                        break
                if exclude == False:
                    r = pos[atom2]-pos[atom1]
                    r2 = np.dot(r,r)
                    # Calculate LJ Force
                    r6 = r2*r2*r2
                    r8 = r6*r2
                    lj_type = self.ityp[atom1]*self.n_types + self.ityp[atom2]
                    lj_type = self.nonbonded_parm_index[lj_type]
                    f = 6/r8*(2*self.lj_params[lj_type,0]/r6-self.lj_params[lj_type,1])*r
                    f_vdw[atom1] -= f
                    f_vdw[atom2] += f
                    # calculate Coulomb Force
                    f = self.charges[atom1]*self.charges[atom2]/np.sqrt(r6)*r
                    f_elec[atom1] -= f
                    f_elec[atom2] += f
        return f_elec, f_vdw

    def hessian(self, pos):
        # Assert that position is of the right size
        assert pos.shape[0] == self.n_atoms, "postition array has incorrect number of atoms"
        # 
        hess_elec = np.zeros((self.n_atoms*3,self.n_atoms*3),dtype=np.float64)
        hess_vdw  = np.zeros((self.n_atoms*3,self.n_atoms*3),dtype=np.float64)
        for atom1 in range(self.n_atoms-1):
            for atom2 in range(atom1+1,self.n_atoms):
                # check excluded atom list
                exclude = False
                offset = np.sum(self.n_excluded_atoms[:atom1])
                for e_atom in range(self.n_excluded_atoms[atom1]):
                    if (self.excluded_atoms_list[offset + e_atom] == atom2):
                        exclude = True
                        break
                if exclude == False:
                    r = pos[atom2]-pos[atom1]
                    r2 = np.dot(r,r)
                    r_mag = np.sqrt(r2)
                    r_norm = r/r_mag
                    # Calculate LJ Hessian
                    r6 = r2*r2*r2
                    r8 = r6*r2
                    lj_type = self.ityp[atom1]*self.n_types + self.ityp[atom2]
                    lj_type = self.nonbonded_parm_index[lj_type]
                    outer = np.outer(r_norm,r_norm)
                    hess_elem = -6/r8*(26*self.lj_params[lj_type,0]/r6-7*self.lj_params[lj_type,1])*outer + 6/r8*(2*self.lj_params[lj_type,0]/r6-self.lj_params[lj_type,1])*(np.identity(3) - outer)
                    hess_vdw[atom1*3:atom1*3+3,atom2*3:atom2*3+3] += hess_elem
                    hess_vdw[atom2*3:atom2*3+3,atom1*3:atom1*3+3] += hess_elem
                    hess_vdw[atom1*3:atom1*3+3,atom1*3:atom1*3+3] -= hess_elem
                    hess_vdw[atom2*3:atom2*3+3,atom2*3:atom2*3+3] -= hess_elem
                    # calculate Coulomb Hessian
                    hess_elem = -self.charges[atom1]*self.charges[atom2]/(r2*r_mag) * ( 3*outer - np.identity(3) )
                    hess_elec[atom1*3:atom1*3+3,atom2*3:atom2*3+3] += hess_elem
                    hess_elec[atom2*3:atom2*3+3,atom1*3:atom1*3+3] += hess_elem
                    hess_elec[atom1*3:atom1*3+3,atom1*3:atom1*3+3] -= hess_elem
                    hess_elec[atom2*3:atom2*3+3,atom2*3:atom2*3+3] -= hess_elem
        return hess_elec, hess_vdw
        
    def numeric_hessian(self, pos, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess_elec = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        hess_vdw = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                pp_e, pp_vdw = self.energy((pos_flat+0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                mm_e, mm_vdw = self.energy((pos_flat-0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                pm_e, pm_vdw = self.energy((pos_flat+0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                mp_e, mp_vdw = self.energy((pos_flat-0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                hess_elec[coord1,coord2] = hess_elec[coord2,coord1] = pp_e + mm_e - pm_e - mp_e 
                hess_vdw[coord1,coord2] = hess_vdw[coord2,coord1] = pp_vdw + mm_vdw - pm_vdw - mp_vdw
        # complete
        hess_elec /= delta2
        hess_vdw /= delta2
        return hess_elec, hess_vdw


class bonds:
    """
    """
    def __init__(self, n_bonds_H, n_bonds_noH, n_types):
        """
        """
        self.n_bonds_H = n_bonds_H
        self.n_bonds_noH = n_bonds_noH
        self.n_bonds = n_bonds_H + n_bonds_noH
        self.n_types = n_types
        self.bond_atoms = np.empty((self.n_bonds,3),dtype=int)
        self.bond_params = np.empty((n_types,2),dtype=np.float64)

    def remove_indeces(self, indeces, n_atoms):
        
        # index to key
        key = indeces_to_key(indeces, n_atoms)
        #
        new_bonds = []
        for bond in range(self.n_bonds):
            if self.bond_atoms[bond,0] not in indeces and self.bond_atoms[bond,1] not in indeces:
                new_bonds.append([key[self.bond_atoms[bond,0]],key[self.bond_atoms[bond,1]],self.bond_atoms[bond,2]])
        # make it an array
        self.bond_atoms = np.array(new_bonds,dtype=int)
        self.n_bonds = self.bond_atoms.shape[0]
        
    def energy(self,pos):
        energy = 0.0
        for bond in range(self.n_bonds):
            r = pos[self.bond_atoms[bond,1]] - pos[self.bond_atoms[bond,0]]
            r_mag = np.linalg.norm(r)
            bond_type = self.bond_atoms[bond,2]
            energy += self.bond_params[bond_type,0] * (r_mag - self.bond_params[bond_type,1])**2
        return energy

    def force(self, pos):
        force = np.zeros(pos.shape, dtype=np.float64)
        for bond in range(self.n_bonds):
            atom1 = self.bond_atoms[bond,0]
            atom2 = self.bond_atoms[bond,1]
            r = pos[atom2] - pos[atom1]
            r_mag = np.linalg.norm(r)
            r_norm = r / r_mag
            bond_type = self.bond_atoms[bond,2]
            f = -2*self.bond_params[bond_type,0] * (r_mag - self.bond_params[bond_type,1])*r_norm
            force[atom1] -= f
            force[atom2] += f
        return force

    def hessian(self, pos):
        # meta data
        n_atoms = pos.shape[0]
        # declare hessian
        hessian = np.zeros((n_atoms*3,n_atoms*3), dtype=np.float64)
        for bond in range(self.n_bonds):
            atom1 = self.bond_atoms[bond,0]
            atom2 = self.bond_atoms[bond,1]
            r = pos[atom2] - pos[atom1]
            r_mag = np.linalg.norm(r)
            r_norm = r / r_mag
            bond_type = self.bond_atoms[bond,2]
            # note the second term is there because cannot guaranteee this component of the potential will be at the minimum
            outer = np.outer(r_norm,r_norm)
            hess_elem = -2*self.bond_params[bond_type,0] * ( np.outer(r_norm,r_norm) +  (r_mag - self.bond_params[bond_type,1])*(np.identity(3) - outer) / r_mag )
            hessian[atom1*3:atom1*3+3,atom2*3:atom2*3+3] += hess_elem
            hessian[atom2*3:atom2*3+3,atom1*3:atom1*3+3] += hess_elem
            hessian[atom1*3:atom1*3+3,atom1*3:atom1*3+3] -= hess_elem
            hessian[atom2*3:atom2*3+3,atom2*3:atom2*3+3] -= hess_elem
        return hessian
        
    def numeric_hessian(self, pos, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                pp = self.energy((pos_flat+0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                mm = self.energy((pos_flat-0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                pm = self.energy((pos_flat+0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                mp = self.energy((pos_flat-0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                hess[coord1,coord2] = hess[coord2,coord1] = pp + mm - pm - mp 
        # complete
        hess /= delta2
        return hess


class angles:
    """
    """
    def __init__(self, n_angles_H, n_angles_noH, n_types):
        """
        """
        self.n_angles_H = n_angles_H
        self.n_angles_noH = n_angles_noH
        self.n_angles = n_angles_H + n_angles_noH
        self.n_types = n_types
        self.angle_atoms = np.empty((self.n_angles,4),dtype=int)
        self.angle_params = np.empty((n_types,2),dtype=np.float64)
        
    def remove_indeces(self, indeces, n_atoms):
        
        # index to key
        key = indeces_to_key(indeces, n_atoms)
        #
        new_angles = []
        for angle in range(self.n_angles):
            if self.angle_atoms[angle,0] not in indeces and self.angle_atoms[angle,1] not in indeces and self.angle_atoms[angle,2] not in indeces :
                new_angles.append([key[self.angle_atoms[angle,0]],key[self.angle_atoms[angle,1]],key[self.angle_atoms[angle,2]],self.angle_atoms[angle,3]])
        # make it an array
        self.angle_atoms = np.array(new_angles,dtype=int)
        self.n_angles = self.angle_atoms.shape[0]

    def energy(self,pos):
        """
        """
        energy = 0.0
        for angle in range(self.n_angles):
            r1 = pos[self.angle_atoms[angle,0]] - pos[self.angle_atoms[angle,1]]
            r1 /= np.linalg.norm(r1)
            r2 = pos[self.angle_atoms[angle,2]] - pos[self.angle_atoms[angle,1]]
            r2 /= np.linalg.norm(r2)
            theta = np.arccos(np.dot(r1,r2))
            angle_type = self.angle_atoms[angle,3]
            energy += self.angle_params[angle_type,0] * (theta - self.angle_params[angle_type,1])**2
        return energy

    def force(self,pos):
        """
        """
        force = np.zeros(pos.shape,dtype=np.float64)
        for angle in range(self.n_angles):
            atom1 = self.angle_atoms[angle,0]
            atom2 = self.angle_atoms[angle,1]
            atom3 = self.angle_atoms[angle,2]
            # compute separation vectors
            r1 = pos[atom1] - pos[atom2]
            r2 = pos[atom2] - pos[atom3]
            # compute dot products
            c11 = np.dot(r1,r1)
            c22 = np.dot(r2,r2)
            c12 = np.dot(r1,r2)
            # compute angle
            theta = np.arccos(-c12/np.sqrt(c11*c22))
            angle_type = self.angle_atoms[angle,3]
            # compute force component common to each
            f = 2*self.angle_params[angle_type,0] * (theta - self.angle_params[angle_type,1])/np.sqrt(c11*c22-c12*c12)
            # add to forces component wise
            force[atom1,0] += f * (c12/c11*r1[0] - r2[0])
            force[atom2,0] += f * ( (1+c12/c22)*r2[0] - (1+c12/c11)*r1[0] )
            force[atom3,0] += f * (r1[0] - c12/c22*r2[0])

            force[atom1,1] += f * (c12/c11*r1[1] - r2[1])
            force[atom2,1] += f * ( (1+c12/c22)*r2[1] - (1+c12/c11)*r1[1] )
            force[atom3,1] += f * (r1[1] - c12/c22*r2[1])

            force[atom1,2] += f * (c12/c11*r1[2] - r2[2])
            force[atom2,2] += f * ( (1+c12/c22)*r2[2] - (1+c12/c11)*r1[2] )
            force[atom3,2] += f * (r1[2] - c12/c22*r2[2])
            
        return force

    def hessian(self,pos, dtype=np.float64):
        """
        """
        # meta data
        n_atoms = pos.shape[0]
        # zero Hessian matrix
        hessian = np.zeros((n_atoms*3,n_atoms*3),dtype=dtype)
        # loop through angles
        for angle in range(self.n_angles):
            # get parameters of this angle
            atomi = self.angle_atoms[angle,0]
            atomj = self.angle_atoms[angle,1]
            atomk = self.angle_atoms[angle,2]
            angle_type = self.angle_atoms[angle,3]
            kij = self.angle_params[angle_type,0]
            theta0 = self.angle_params[angle_type,1]
            # compute separation vectors
            rij = (pos[atomi] - pos[atomj]).astype(dtype)
            rkj = (pos[atomk] - pos[atomj]).astype(dtype)
            # normalize
            rij_mag = np.linalg.norm(rij)
            rkj_mag = np.linalg.norm(rkj)
            rij_hat = rij/rij_mag
            rkj_hat = rkj/rkj_mag
            # compute cos and sin of angle            
            cos_theta = np.dot(rij_hat,rkj_hat)
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)
            # first derivative vectors
            dtheta_dri = (cos_theta*rij_hat - rkj_hat)/(np.sqrt(1-cos_theta**2)*rij_mag)
            dtheta_drk = (cos_theta*rkj_hat - rij_hat)/(np.sqrt(1-cos_theta**2)*rkj_mag)
            dtheta_drj = -dtheta_dri - dtheta_drk
            # first and second derivative scalars
            dE_dtheta = 2*kij*(theta - theta0)
            d2E_dtheta2 = 2*kij
            # compute outer products
            rij_rij = np.outer(rij_hat,rij_hat)
            rkj_rkj = np.outer(rkj_hat,rkj_hat)
            rij_rkj = np.outer(rij_hat,rkj_hat)
            rkj_rij = rij_rkj.T # np.outer(rkj_hat,rij_hat)
            I3 = np.identity(3,dtype=dtype)
            # Hessian for atom i with atom i
            T1 = d2E_dtheta2*np.outer(dtheta_dri,dtheta_dri)
            T2 = rij_rkj + cos_theta * ( cos_theta * rkj_rij - rkj_rkj - rij_rij )
            T2 /= sin_theta**3
            T3 = rkj_rij + cos_theta * ( I3 - 2*rij_rij)
            T3 /= np.sqrt(1-cos_theta**2)  # np.abs(sin_theta)
            hessian[atomi*3:atomi*3+3,atomi*3:atomi*3+3] += T1 + dE_dtheta * ( T2 + T3 ) /rij_mag**2
            # Hessian for atom j with atom i
            T1 = np.outer( rij_hat/rij_mag - cos_theta/sin_theta*dtheta_drj, dtheta_dri)
            T2 = cos_theta/rij_mag * (rij_rij - I3) - sin_theta*np.outer(dtheta_drj,rij_hat) - (rkj_rkj - I3)/rkj_mag
            T2 /= (rij_mag*np.sqrt(1-cos_theta**2))
            d2theta_drj_dri = T1 + T2
            hess_elem = d2E_dtheta2*np.outer(dtheta_drj,dtheta_dri) + dE_dtheta * d2theta_drj_dri
            hessian[atomj*3:atomj*3+3,atomi*3:atomi*3+3] += hess_elem
            hessian[atomi*3:atomi*3+3,atomj*3:atomj*3+3] += hess_elem.T
            # Hessian for atom k with atom i
            T1 = -cos_theta/sin_theta*np.outer(dtheta_drk,dtheta_dri)
            T2 = -sin_theta/np.sqrt(1-cos_theta**2)*np.outer(dtheta_drk,rij_hat)/rij_mag
            T3 = rkj_rkj - I3
            T3 /= (rij_mag*rkj_mag*np.sqrt(1-cos_theta**2))
            hess_elem = d2E_dtheta2*np.outer(dtheta_drk,dtheta_dri) + dE_dtheta * (T1+T2+T3)
            hessian[atomk*3:atomk*3+3,atomi*3:atomi*3+3] += hess_elem
            hessian[atomi*3:atomi*3+3,atomk*3:atomk*3+3] += hess_elem.T
            # Hessian for atom j with atom k
            T1 = np.outer( rkj_hat/rkj_mag - cos_theta/sin_theta*dtheta_drj, dtheta_drk)
            T2 = cos_theta/rkj_mag * (rkj_rkj - I3) - sin_theta*np.outer(dtheta_drj,rkj_hat) - (rij_rij - I3)/rij_mag
            T2 /= (rkj_mag*np.sqrt(1-cos_theta**2))
            d2theta_drj_drk = T1 + T2
            hess_elem = d2E_dtheta2*np.outer(dtheta_drj,dtheta_drk) + dE_dtheta * d2theta_drj_drk
            hessian[atomj*3:atomj*3+3,atomk*3:atomk*3+3] += hess_elem
            hessian[atomk*3:atomk*3+3,atomj*3:atomj*3+3] += hess_elem.T
            # Hessian for atom k with atom k
            T1 = -np.outer(sin_theta*rkj_hat + rkj_mag*cos_theta*dtheta_drk,dtheta_drk)
            T1 /= rkj_mag*sin_theta
            T2 = cos_theta/rkj_mag*(I3-rkj_rkj) - sin_theta*np.outer(dtheta_drk,rkj_hat)
            T2 /= rkj_mag*np.sqrt(1-cos_theta**2)  
            hessian[atomk*3:atomk*3+3,atomk*3:atomk*3+3] += d2E_dtheta2*np.outer(dtheta_drk,dtheta_drk) + dE_dtheta*(T1+T2)
            # Hessian for atom j wth atom j
            hessian[atomj*3:atomj*3+3,atomj*3:atomj*3+3] += d2E_dtheta2*np.outer(dtheta_drj,dtheta_drj) - dE_dtheta * (d2theta_drj_dri + d2theta_drj_drk)
            
        return hessian
    
    def numeric_hessian(self, pos, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                pp = self.energy((pos_flat+0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                mm = self.energy((pos_flat-0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                pm = self.energy((pos_flat+0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                mp = self.energy((pos_flat-0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                hess[coord1,coord2] = hess[coord2,coord1] = pp + mm - pm - mp 
        # complete
        hess /= delta2
        return hess

# some constants we will use
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

class dihedrals:
    """
    """
    def __init__(self, n_dihedrals_H, n_dihedrals_noH, n_types):
        """
        """
        self.n_dihedrals_H = n_dihedrals_H 
        self.n_dihedrals_noH = n_dihedrals_noH
        self.n_dihedrals = n_dihedrals_H + n_dihedrals_noH
        self.n_types = n_types
        self.dihedral_atoms = np.empty((self.n_dihedrals,5),dtype=int)
        self.dihedral_params = np.empty((n_types,3), dtype=np.float64)
        self.scaled_14_factors = np.empty((n_types,2), dtype=np.float64)

    def remove_indeces(self, indeces, n_atoms):
        
        # index to key
        key = indeces_to_key(indeces, n_atoms)
        #
        new_dihedrals = []
        for dih in range(self.n_dihedrals):
            if self.dihedral_atoms[dih,0] not in indeces and self.dihedral_atoms[dih,1] not in indeces and abs(self.dihedral_atoms[dih,2]) not in indeces and abs(self.dihedral_atoms[dih,3]) not in indeces:
                new_dihedrals.append([key[self.dihedral_atoms[dih,0]],key[self.dihedral_atoms[dih,1]],np.sign(self.dihedral_atoms[dih,2])*key[abs(self.dihedral_atoms[dih,2])],np.sign(self.dihedral_atoms[dih,3])*key[abs(self.dihedral_atoms[dih,3])],self.dihedral_atoms[dih,4]])
        # make it an array
        self.dihedral_atoms = np.array(new_dihedrals,dtype=int)
        self.n_dihedrals = self.dihedral_atoms.shape[0]
        
        
    def energy(self,pos,atoms):
        """
        """
        dih_energy = 0.0
        s14_e = 0.0
        s14_vdw = 0.0
        for dih in range(self.n_dihedrals):
            atom1 = self.dihedral_atoms[dih,0]
            atom2 = self.dihedral_atoms[dih,1]
            atom3 = self.dihedral_atoms[dih,2]
            atom4 = self.dihedral_atoms[dih,3]
            dih_type = self.dihedral_atoms[dih,4]
            # if the third or fourth atom numbers are negative we don't want to compute scaled 1-4 
            if atom3 > 0 and atom4 > 0:
                r = pos[atom4]-pos[atom1]
                r_mag = np.linalg.norm(r)
                # Calculate LJ Energy
                r_mag6 = r_mag**6
                if atom1 < atom4:
                    lj_type = atoms.ityp[atom1]*atoms.n_types + atoms.ityp[atom4]
                else: 
                    lj_type = atoms.ityp[atom4]*atoms.n_types + atoms.ityp[atom1]
                lj_type = atoms.nonbonded_parm_index[lj_type]
                # add scaled LJ energy
                s14_vdw += self.scaled_14_factors[dih_type,1]*(atoms.lj_params[lj_type,0]/r_mag6**2 - atoms.lj_params[lj_type,1]/r_mag6)
                # add scaled Coulomb Energy
                s14_e += self.scaled_14_factors[dih_type,0]*atoms.charges[atom1]*atoms.charges[atom4]/r_mag
            atom3 = abs(atom3)
            atom4 = abs(atom4)   
            phi = calc_dihedrals(pos[atom1], pos[atom2], pos[atom3], pos[atom4])
            dih_energy += self.dihedral_params[dih_type,0]*( 1 + np.cos(self.dihedral_params[dih_type,1] * phi - self.dihedral_params[dih_type,2] ))
        return dih_energy, s14_e, s14_vdw

    def force(self,pos,atoms):
        """
        """
        f_dih = np.zeros(pos.shape,dtype=np.float64)
        f_s14_e = np.zeros(pos.shape,dtype=np.float64)
        f_s14_vdw = np.zeros(pos.shape,dtype=np.float64)
        for dih in range(self.n_dihedrals):
            atom1 = self.dihedral_atoms[dih,0]
            atom2 = self.dihedral_atoms[dih,1]
            atom3 = self.dihedral_atoms[dih,2]
            atom4 = self.dihedral_atoms[dih,3]
            dih_type = self.dihedral_atoms[dih,4]
            # if the third or fourth atom numbers are negative we don't want to compute scaled 1-4 
            if atom3 > 0 and atom4 > 0:
                r = pos[atom4]-pos[atom1]
                r2 = np.dot(r,r)
                r6 = r2*r2*r2
                r8 = r6*r2
                # Calculate LJ Energy
                if atom1 < atom4:
                    lj_type = atoms.ityp[atom1]*atoms.n_types + atoms.ityp[atom4]
                else: 
                    lj_type = atoms.ityp[atom4]*atoms.n_types + atoms.ityp[atom1]
                lj_type = atoms.nonbonded_parm_index[lj_type]
                # add scaled LJ force
                f = self.scaled_14_factors[dih_type,1]*6/r8*(2*atoms.lj_params[lj_type,0]/r6-atoms.lj_params[lj_type,1])*r
                f_s14_vdw[atom1] -= f
                f_s14_vdw[atom4] += f
                # add scaled Coulomb force
                f = self.scaled_14_factors[dih_type,0]*atoms.charges[atom1]*atoms.charges[atom4]/np.sqrt(r6)*r
                f_s14_e[atom1] -= f
                f_s14_e[atom4] += f
            atom3 = abs(atom3)
            atom4 = abs(atom4)   
            phi = calc_dihedrals(pos[atom1], pos[atom2], pos[atom3], pos[atom4])
            # calculate dihedral force
            # separation vectors
            # from https://salilab.org/modeller/9v5/manual/node435.html
            # They cite: van Gunsteren 1993 J Mol Biol v234 p751-762
            rij = pos[atom1] - pos[atom2]
            rkj = pos[atom3] - pos[atom2]
            rkl = pos[atom3] - pos[atom4]
            # cross products
            rmj = np.cross(rij,rkj)
            rnk = np.cross(rkj,rkl)
            # some magnitudes etc
            rkj_2 = np.dot(rkj,rkj)
            rkj_mag = np.sqrt(rkj_2)
            rmj_2 = np.dot(rmj,rmj)
            rnk_2 = np.dot(rnk,rnk)
            a = np.dot(rij,rkj)/rkj_2
            b = np.dot(rkl,rkj)/rkj_2
            # -dE/dphi:
            f = self.dihedral_params[dih_type,0] * self.dihedral_params[dih_type,1] * np.sin(self.dihedral_params[dih_type,1]*phi-self.dihedral_params[dih_type,2])
            # now force components
            f1 = rkj_mag/rmj_2*rmj
            f4 = -rkj_mag/rnk_2*rnk
            f_dih[atom1] += f * f1
            f_dih[atom2] += f * ( (a-1)*f1 - b*f4 ) 
            f_dih[atom3] += f * ( -a*f1 + (b-1)*f4 ) 
            f_dih[atom4] += f * f4
 
            
        return f_dih, f_s14_e, f_s14_vdw

    def hessian(self,pos,atoms):
        """
        """
        # meta data
        n_atoms = pos.shape[0]
        hessian_dih = np.zeros((n_atoms*3,n_atoms*3),dtype=np.float64)
        hessian_vdw = np.zeros((n_atoms*3,n_atoms*3),dtype=np.float64)
        hessian_elec = np.zeros((n_atoms*3,n_atoms*3),dtype=np.float64)
        for dih in range(self.n_dihedrals):
            atom1 = self.dihedral_atoms[dih,0]
            atom2 = self.dihedral_atoms[dih,1]
            atom3 = self.dihedral_atoms[dih,2]
            atom4 = self.dihedral_atoms[dih,3]
            dih_type = self.dihedral_atoms[dih,4]
            # dihedral parameters
            V = self.dihedral_params[dih_type,0]
            n = self.dihedral_params[dih_type,1]
            d = self.dihedral_params[dih_type,2]
            # s14
            # if either the third or fourth atom numbers are negative we don't want to compute scaled 1-4 
            if atom3 > 0 and atom4 > 0:
                r = pos[atom4]-pos[atom1]
                r2 = np.dot(r,r)
                r_mag = np.sqrt(r2)
                r_norm = r/r_mag
                # Calculate LJ Hessian
                r6 = r2*r2*r2
                r8 = r6*r2
                # get LJ type
                if atom1 < atom4:
                    lj_type = atoms.ityp[atom1]*atoms.n_types + atoms.ityp[atom4]
                else: 
                    lj_type = atoms.ityp[atom4]*atoms.n_types + atoms.ityp[atom1]
                lj_type = atoms.nonbonded_parm_index[lj_type]
                # LJ Hessian element
                outer = np.outer(r_norm,r_norm)
                I3 = np.identity(3,dtype=np.float64)
                hess_elem = 6*self.scaled_14_factors[dih_type,1]/r8*( -(26*atoms.lj_params[lj_type,0]/r6-7*atoms.lj_params[lj_type,1])*outer + (2*atoms.lj_params[lj_type,0]/r6-atoms.lj_params[lj_type,1])*(I3 - outer))
                hessian_vdw[atom1*3:atom1*3+3,atom4*3:atom4*3+3] += hess_elem
                hessian_vdw[atom4*3:atom4*3+3,atom1*3:atom1*3+3] += hess_elem
                hessian_vdw[atom1*3:atom1*3+3,atom1*3:atom1*3+3] -= hess_elem
                hessian_vdw[atom4*3:atom4*3+3,atom4*3:atom4*3+3] -= hess_elem
                # calculate Coulomb Hessian
                hess_elem = -self.scaled_14_factors[dih_type,0] * atoms.charges[atom1]*atoms.charges[atom4]/(r2*r_mag) * ( 3*outer - I3 )
                hessian_elec[atom1*3:atom1*3+3,atom4*3:atom4*3+3] += hess_elem
                hessian_elec[atom4*3:atom4*3+3,atom1*3:atom1*3+3] += hess_elem
                hessian_elec[atom1*3:atom1*3+3,atom1*3:atom1*3+3] -= hess_elem
                hessian_elec[atom4*3:atom4*3+3,atom4*3:atom4*3+3] -= hess_elem
            atom3 = abs(atom3)
            atom4 = abs(atom4)   
            # seperation vectors
            rij = pos[atom1] - pos[atom2]
            rkj = pos[atom3] - pos[atom2]
            rkl = pos[atom3] - pos[atom4]
            # magnitudes etc
            rkj_2 = np.dot(rkj,rkj)
            rkj_mag = np.sqrt(rkj_2)
            rkj_hat = rkj/rkj_mag
            # cross products
            rmj = np.cross(rij,rkj)
            rmj_2 = np.dot(rmj,rmj)
            rmj_mag = np.sqrt(rmj_2)
            rmj_hat = rmj / rmj_mag
            rnk = np.cross(rkj,rkl)
            rnk_2 = np.dot(rnk,rnk)
            rnk_mag = np.sqrt(rnk_2)
            rnk_hat = rnk / rnk_mag
            # compute phi
            phi = np.sign(np.dot(rkj,np.cross(rmj,rnk)))*np.arccos(np.dot(rmj_hat,rnk_hat))
            # Derivatives of E w.r.t. phi
            dE_dphi = -V*n*np.sin(n*phi-d)
            d2E_dphi2 = -V*n*n*np.cos(n*phi-d)
            # First derivatives of phi w.r.t. coordinates
            dphi_dri = rkj_mag/rmj_2*rmj
            dphi_drl = -rkj_mag/rnk_2*rnk
            dphi_drj = (np.dot(rij,rkj)/rkj_2 - 1)*dphi_dri - np.dot(rkl,rkj)/rkj_2*dphi_drl
            dphi_drk = (np.dot(rkl,rkj)/rkj_2 - 1)*dphi_drl - np.dot(rij,rkj)/rkj_2*dphi_dri
            # ii term
            drmjT_dri = -np.dot(eijk,rkj)
            drmj_mag_dri = -np.cross(rmj_hat,rkj)
            d2phi_dri_dri = -rkj_mag/rmj_2*(2*np.outer(drmj_mag_dri,rmj_hat) - drmjT_dri)
            hess_elem = d2E_dphi2*np.outer(dphi_dri,dphi_dri) + dE_dphi*d2phi_dri_dri
            hessian_dih[atom1*3:atom1*3+3,atom1*3:atom1*3+3] += hess_elem
            # ll term
            drnkT_drl = -np.dot(eijk,rkj)
            drnk_mag_drl = -np.cross(rnk_hat,rkj)
            d2phi_drl_drl = rkj_mag/rnk_2*(2*np.outer(drnk_mag_drl ,rnk_hat) - drnkT_drl)
            hess_elem = d2E_dphi2*np.outer(dphi_drl,dphi_drl) + dE_dphi*d2phi_drl_drl
            hessian_dih[atom4*3:atom4*3+3,atom4*3:atom4*3+3] += hess_elem
            # ji term
            rki = pos[atom3] - pos[atom1]
            drmjT_drj = np.dot(eijk,rki)
            drmj_mag_drj = np.cross(rmj_hat,rki)
            d2phi_drj_dri = -rkj_mag/rmj_2*(2*np.outer(drmj_mag_drj ,rmj_hat) - drmjT_drj) - np.outer(rkj_hat,rmj_hat)/rmj_mag
            hess_elem = d2E_dphi2*np.outer(dphi_drj,dphi_dri) + dE_dphi*d2phi_drj_dri
            hessian_dih[atom2*3:atom2*3+3,atom1*3:atom1*3+3] += hess_elem
            hessian_dih[atom1*3:atom1*3+3,atom2*3:atom2*3+3] += hess_elem.T
            # ki term
            drmjT_drk = np.dot(eijk,rij)
            drmj_mag_drk = np.cross(rmj_hat,rij)
            d2phi_drk_dri = -rkj_mag/rmj_2*(2*np.outer(drmj_mag_drk ,rmj_hat) - drmjT_drk) + np.outer(rkj_hat,rmj_hat)/rmj_mag
            hess_elem = d2E_dphi2*np.outer(dphi_drk,dphi_dri) + dE_dphi*d2phi_drk_dri
            hessian_dih[atom3*3:atom3*3+3,atom1*3:atom1*3+3] += hess_elem
            hessian_dih[atom1*3:atom1*3+3,atom3*3:atom3*3+3] += hess_elem.T
            # jl term
            drnkT_drj = np.dot(eijk,rkl)
            drnk_mag_drj = np.cross(rnk_hat,rkl)
            d2phi_drj_drl = rkj_mag/rnk_2*(2*np.outer(drnk_mag_drj ,rnk_hat) - drnkT_drj) + np.outer(rkj_hat,rnk_hat)/rnk_mag
            hess_elem = d2E_dphi2*np.outer(dphi_drj,dphi_drl) + dE_dphi*d2phi_drj_drl
            hessian_dih[atom2*3:atom2*3+3,atom4*3:atom4*3+3] += hess_elem
            hessian_dih[atom4*3:atom4*3+3,atom2*3:atom2*3+3] += hess_elem.T
            # kl term
            rjl = pos[atom2] - pos[atom4]
            drnkT_drk = -np.dot(eijk,rjl)
            drnk_mag_drk = -np.cross(rnk_hat,rjl)
            d2phi_drk_drl = rkj_mag/rnk_2*(2*np.outer(drnk_mag_drk ,rnk_hat) - drnkT_drk) - np.outer(rkj_hat,rnk_hat)/rnk_mag
            hess_elem = d2E_dphi2*np.outer(dphi_drk,dphi_drl) + dE_dphi*d2phi_drk_drl
            hessian_dih[atom3*3:atom3*3+3,atom4*3:atom4*3+3] += hess_elem
            hessian_dih[atom4*3:atom4*3+3,atom3*3:atom3*3+3] += hess_elem.T
            # li term
            hess_elem = d2E_dphi2*np.outer(dphi_drl,dphi_dri)
            hessian_dih[atom4*3:atom4*3+3,atom1*3:atom1*3+3] += hess_elem
            hessian_dih[atom1*3:atom1*3+3,atom4*3:atom4*3+3] += hess_elem.T
            # jj term
            A = 2*np.dot(rij,rkj)*rkj_hat/rkj_mag**3 - (rij+rkj)/rkj_2
            B = 2*np.dot(rkl,rkj)*rkj_hat/rkj_mag**3 - rkl/rkj_2
            d2phi_drj2 = np.outer(A,dphi_dri) + (np.dot(rij,rkj)/rkj_2 - 1)*d2phi_drj_dri - np.outer(B,dphi_drl) - np.dot(rkl,rkj)/rkj_2*d2phi_drj_drl
            hess_elem = d2E_dphi2*np.outer(dphi_drj,dphi_drj) + dE_dphi*d2phi_drj2
            hessian_dih[atom2*3:atom2*3+3,atom2*3:atom2*3+3] += hess_elem
            # kk term 
            A = -2*np.dot(rkl,rkj)/rkj_mag**3*rkj_hat + (rkj+rkl)/rkj_2
            B = -2*np.dot(rij,rkj)/rkj_mag**3*rkj_hat + rij/rkj_2
            d2phi_drk2 = np.outer(A,dphi_drl) + (np.dot(rkl,rkj)/rkj_2 - 1)*d2phi_drk_drl - np.outer(B,dphi_dri) - np.dot(rij,rkj)/rkj_2*d2phi_drk_dri
            hess_elem = d2E_dphi2*np.outer(dphi_drk,dphi_drk) + dE_dphi*d2phi_drk2
            hessian_dih[atom3*3:atom3*3+3,atom3*3:atom3*3+3] += hess_elem
            # jk term
            A = 2*np.dot(rkl,rkj)*rkj_hat/rkj_mag**3 - rkl/rkj_2
            B = 2*np.dot(rij,rkj)*rkj_hat/rkj_mag**3 - (rij+rkj)/rkj_2
            d2phi_drj_drk = np.outer(A,dphi_drl) + (np.dot(rkl,rkj)/rkj_2 - 1)*d2phi_drj_drl - np.outer(B,dphi_dri) - np.dot(rij,rkj)/rkj_2*d2phi_drj_dri
            hess_elem = d2E_dphi2*np.outer(dphi_drj,dphi_drk) + dE_dphi*d2phi_drj_drk
            hessian_dih[atom2*3:atom2*3+3,atom3*3:atom3*3+3] += hess_elem
            hessian_dih[atom3*3:atom3*3+3,atom2*3:atom2*3+3] += hess_elem.T
            
        return hessian_dih, hessian_elec, hessian_vdw

    def numeric_hessian(self, pos, atoms, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess_dih = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        hess_elec = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        hess_vdw = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                pp, pp_e, pp_vdw = self.energy((pos_flat+0.5*(coord1_delta+coord2_delta)).reshape(-1,3), atoms)
                mm, mm_e, mm_vdw = self.energy((pos_flat-0.5*(coord1_delta+coord2_delta)).reshape(-1,3), atoms)
                pm, pm_e, pm_vdw = self.energy((pos_flat+0.5*(coord1_delta-coord2_delta)).reshape(-1,3), atoms)
                mp, mp_e, mp_vdw = self.energy((pos_flat-0.5*(coord1_delta-coord2_delta)).reshape(-1,3), atoms)
                hess_dih[coord1,coord2] = hess_dih[coord2,coord1] = pp + mm - pm - mp 
                hess_elec[coord1,coord2] = hess_elec[coord2,coord1] = pp_e + mm_e - pm_e - mp_e 
                hess_vdw[coord1,coord2] = hess_vdw[coord2,coord1] = pp_vdw + mm_vdw - pm_vdw - mp_vdw 
        # complete
        hess_dih /= delta2
        hess_elec /= delta2
        hess_vdw /= delta2
        return hess_dih, hess_elec, hess_vdw

class cmap:
    """
    """
    def __init__(self, cmap_term_count, cmap_type_count):
        """
        """
        self.term_count = cmap_term_count
        self.type_count = cmap_type_count
        self.resolution = np.empty(self.term_count,dtype=int)
        self.bin_size = np.empty(self.term_count,dtype=np.float64)
        self.atom_list = np.empty((self.term_count,6),dtype=int)
        # start a counter
        self.parameter_count = 0
        # set some other defaults
        self.phi_min = -np.pi
        self.psi_min = -np.pi
        self.phi_max = np.pi
        self.psi_max = np.pi

    def remove_indeces(self, indeces, n_atoms):
        
        # index to key
        key = indeces_to_key(indeces, n_atoms)
        #
        new_cmaps = []
        for cmap in range(self.term_count):
            if self.atom_list[cmap,0] not in indeces and self.atom_list[cmap,1] not in indeces and self.atom_list[cmap,2] not in indeces and self.atom_list[cmap,3] not in indeces and self.atom_list[cmap,4] not in indeces:
                new_cmaps.append([key[self.atom_list[cmap,0]],key[self.atom_list[cmap,1]],key[self.atom_list[cmap,2]],key[self.atom_list[cmap,3]],key[self.atom_list[cmap,4]],self.atom_list[cmap,5]])
        # make it an array
        self.atom_list = np.array(new_cmaps,dtype=int)
        self.term_count = self.atom_list.shape[0]

    def allocate_parameter_list(self):

        self.parameter_list = []
        for i in range(self.term_count):
            self.parameter_list.append(np.empty((self.resolution[i],self.resolution[i]),dtype=np.float64))
            #
            self.bin_size[i] = 2*np.pi/self.resolution[i]

    def fit_interpolators(self,buffer = 12):
        """
        """

        self.interp = []
        for i in range(self.term_count):
            phis = np.arange(-np.pi-buffer*self.bin_size[i],np.pi+buffer*self.bin_size[i],self.bin_size[i],dtype=np.float64)
            psis = np.arange(-np.pi-buffer*self.bin_size[i],np.pi+buffer*self.bin_size[i],self.bin_size[i],dtype=np.float64)
            new_parameter_list = np.empty((self.resolution[i]+2*buffer,self.resolution[i]+2*buffer),dtype=np.float64)
            new_parameter_list[buffer:-buffer,buffer:-buffer] = self.parameter_list[i]
            # pad columns
            new_parameter_list[:,:buffer] = new_parameter_list[:,-2*buffer:-buffer]
            new_parameter_list[:,-buffer:] = new_parameter_list[:,buffer:2*buffer]
            # pad rows
            new_parameter_list[:buffer,:] = new_parameter_list[-2*buffer:-buffer,:]
            new_parameter_list[-buffer:,:] = new_parameter_list[buffer:2*buffer,:]
            # fit bicubic spline interpolator
            self.interp.append(RegularGridInterpolator((phis,psis),new_parameter_list, method="cubic", bounds_error=False, fill_value=None))

    def energy(self,pos):
        """
        """
        cmap_energy = 0.0
        for cmap in range(self.term_count):
            atom1 = self.atom_list[cmap,0]
            atom2 = self.atom_list[cmap,1]
            atom3 = self.atom_list[cmap,2]
            atom4 = self.atom_list[cmap,3]
            atom5 = self.atom_list[cmap,4]
            cmap_type = self.atom_list[cmap,5]
            phi = calc_dihedrals(pos[atom1], pos[atom2], pos[atom3], pos[atom4])
            psi = calc_dihedrals(pos[atom2], pos[atom3], pos[atom4], pos[atom5])
            #print(phi*180/np.pi, psi*180/np.pi)
            cmap_energy += self.interp[cmap_type]((phi, psi))
        return cmap_energy

    def force(self,pos):
        """
        """
        f_cmap = np.empty(pos.shape,dtype=np.float64)
        for cmap in range(self.term_count):
            atom1 = self.atom_list[cmap,0]
            atom2 = self.atom_list[cmap,1]
            atom3 = self.atom_list[cmap,2]
            atom4 = self.atom_list[cmap,3]
            atom5 = self.atom_list[cmap,4]
            cmap_type = self.atom_list[cmap,5]
            phi = calc_dihedrals(pos[atom1], pos[atom2], pos[atom3], pos[atom4])
            psi = calc_dihedrals(pos[atom2], pos[atom3], pos[atom4], pos[atom5])
            # for phi:
            # from https://salilab.org/modeller/9v5/manual/node435.html
            rij = pos[atom1] - pos[atom2]
            rkj = pos[atom3] - pos[atom2]
            rkl = pos[atom3] - pos[atom4]
            # cross products
            rmj = np.cross(rij,rkj)
            rnk = np.cross(rkj,rkl)
            # some magnitudes etc
            rkj_2 = np.dot(rkj,rkj)
            rkj_mag = np.sqrt(rkj_2)
            rmj_2 = np.dot(rmj,rmj)
            rnk_2 = np.dot(rnk,rnk)
            a = np.dot(rij,rkj)/rkj_2
            b = np.dot(rkl,rkj)/rkj_2
            # -dE/dphi:
            f = -self.interp[cmap_type]((phi, psi),nu=(1,0))
            # now force components
            f1 = rkj_mag/rmj_2*rmj
            f4 = -rkj_mag/rnk_2*rnk
            f_cmap[atom1] += f * f1
            f_cmap[atom2] += f * ( (a-1)*f1 - b*f4 )
            f_cmap[atom3] += f * ( -a*f1 + (b-1)*f4 )
            f_cmap[atom4] += f * f4
            # for psi:
            # from https://salilab.org/modeller/9v5/manual/node435.html
            rij = pos[atom2] - pos[atom3]
            rkj = pos[atom4] - pos[atom3]
            rkl = pos[atom4] - pos[atom5]
            # cross products
            rmj = np.cross(rij,rkj)
            rnk = np.cross(rkj,rkl)
            # some magnitudes etc
            rkj_2 = np.dot(rkj,rkj)
            rkj_mag = np.sqrt(rkj_2)
            rmj_2 = np.dot(rmj,rmj)
            rnk_2 = np.dot(rnk,rnk)
            a = np.dot(rij,rkj)/rkj_2
            b = np.dot(rkl,rkj)/rkj_2
            # -dE/dpsi:
            f = -self.interp[cmap_type]((phi, psi),nu=(0,1))
            # now force components
            f1 = rkj_mag/rmj_2*rmj
            f4 = -rkj_mag/rnk_2*rnk
            f_cmap[atom2] += f * f1
            f_cmap[atom3] += f * ( (a-1)*f1 - b*f4 )
            f_cmap[atom4] += f * ( -a*f1 + (b-1)*f4 )
            f_cmap[atom5] += f * f4

        return f_cmap

    def hessian(self,pos, delta=1e-3):
        """
        """
        n_atoms = pos.shape[0]
        hess_cmap = np.zeros((n_atoms*3,n_atoms*3),dtype=np.float64)
        for cmap in range(self.term_count):
            atom_i, atom_j, atom_k, atom_l, atom_m, cmap_type = self.atom_list[cmap]
            phi = calc_dihedrals(pos[atom_i], pos[atom_j], pos[atom_k], pos[atom_l])
            psi = calc_dihedrals(pos[atom_j], pos[atom_k], pos[atom_l], pos[atom_m])

            dE_dphi = self.interp[cmap_type]((phi, psi), nu=(1, 0))
            dE_dpsi = self.interp[cmap_type]((phi, psi), nu=(0, 1))
            d2E_dphi2 = self.interp[cmap_type]((phi, psi), nu=(2, 0))
            d2E_dpsi2 = self.interp[cmap_type]((phi, psi), nu=(0, 2))
            d2E_dphidpsi = self.interp[cmap_type]((phi, psi), nu=(1, 1))

            print(f"phi: {phi:.4f}, psi: {psi:.4f}")
            print(f"d2E_dphi2: {d2E_dphi2:.6f}, d2E_dpsi2: {d2E_dpsi2:.6f}, d2E_dphidpsi: {d2E_dphidpsi:.6f}")

            dphi, d2phi = compute_dihedral_derivatives(pos[atom_i], pos[atom_j], pos[atom_k], pos[atom_l], second_derivatives=True, eps=delta)
            dpsi, d2psi = compute_dihedral_derivatives(pos[atom_j], pos[atom_k], pos[atom_l], pos[atom_m], second_derivatives=True, eps=delta)

            dphi_dr = [dphi.get(i, np.zeros(3)) for i in [0, 1, 2, 3]] + [np.zeros(3)]
            dpsi_dr = [np.zeros(3)] + [dpsi.get(i, np.zeros(3)) for i in [0, 1, 2, 3]]

            for a in range(5):
                for b in range(a, 5):
                    i = self.atom_list[cmap, a]
                    j = self.atom_list[cmap, b]

                    H = d2E_dphi2 * np.outer(dphi_dr[a], dphi_dr[b]) \
                        + d2E_dpsi2 * np.outer(dpsi_dr[a], dpsi_dr[b]) \
                        + d2E_dphidpsi * (np.outer(dphi_dr[a], dpsi_dr[b]) + np.outer(dpsi_dr[a], dphi_dr[b])) \
                        + dE_dphi * d2phi.get((a, b), np.zeros((3, 3))) \
                        + dE_dpsi * d2psi.get((a, b), np.zeros((3, 3)))

                    hess_cmap[i*3:i*3+3, j*3:j*3+3] += H
                    if i != j:
                        hess_cmap[j*3:j*3+3, i*3:i*3+3] += H.T

        return hess_cmap


    def numeric_hessian(self, pos, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess_cmap = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                pp = self.energy((pos_flat+0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                mm = self.energy((pos_flat-0.5*(coord1_delta+coord2_delta)).reshape(-1,3))
                pm = self.energy((pos_flat+0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                mp = self.energy((pos_flat-0.5*(coord1_delta-coord2_delta)).reshape(-1,3))
                hess_cmap[coord1,coord2] = hess_cmap[coord2,coord1] = pp + mm - pm - mp
        # complete
        hess_cmap /= delta2
        return hess_cmap

class mm:
    """
    """
    def __init__(self, parm7_file_name):
        self.parm7_file_name = parm7_file_name
        self._read_parm7()

    def remove_indeces(self, indeces):
        
        old_n_atoms = self.atoms.n_atoms
        self.atoms.remove_indeces(indeces)
        self.bonds.remove_indeces(indeces, old_n_atoms)
        self.angles.remove_indeces(indeces, old_n_atoms)
        self.dihedrals.remove_indeces(indeces, old_n_atoms)
        self.cmap.remove_indeces(indeces, old_n_atoms)
    
    def energy(self, pos):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"

        self.bond_energy = self.bonds.energy(pos)
        self.angle_energy = self.angles.energy(pos)
        self.dihedral_energy, self.s14_elec, self.s14_vdw = self.dihedrals.energy(pos,self.atoms)
        self.elec_energy, self.vdw_energy = self.atoms.energy(pos)
        self.cmap_energy = self.cmap.energy(pos)
        print("Bond energy:", self.bond_energy)
        print("Angle energy:", self.angle_energy)
        print("Dihedral energy:", self.dihedral_energy)
        print("Electrostatic energy:", self.elec_energy)
        print("VDW energy:", self.vdw_energy)
        print("Scaled 1-4 vdw:", self.s14_vdw)
        print("Scaled 1-4 electrostatics:", self.s14_elec)
        print("CMAP energy:", self.cmap_energy)
        self.total_energy = self.bond_energy + self.angle_energy + self.dihedral_energy + self.elec_energy + self.vdw_energy+ self.s14_elec + self.s14_vdw + self.cmap_energy
        print("Total energy:", self.total_energy)
        return self.total_energy

    def force(self, pos):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"

        self.bond_force = self.bonds.force(pos)
        self.angle_force = self.angles.force(pos)
        self.dihedral_force, self.s14_elec_force, self.s14_vdw_force = self.dihedrals.force(pos,self.atoms)
        self.elec_force, self.vdw_force = self.atoms.force(pos)
        self.cmap_force = self.cmap.force(pos)
        self.total_force = self.bond_force + self.angle_force + self.elec_force + self.vdw_force + self.dihedral_force + self.s14_elec_force + self.s14_vdw_force + self.cmap.force(pos)
        #print("Total Force:", self.total_force)
        #return total_force

    def hessian(self, pos, delta=1e-3):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"

        self.bond_hessian = self.bonds.hessian(pos)
        self.angle_hessian = self.angles.hessian(pos)
        self.dihedral_hessian, self.s14_elec_hessian, self.s14_vdw_hessian = self.dihedrals.hessian(pos,self.atoms)
        self.elec_hessian, self.vdw_hessian = self.atoms.hessian(pos)
        self.cmap_hessian = self.cmap.numeric_hessian(pos, delta=delta)
        self.total_hessian = self.bond_hessian + self.angle_hessian + self.elec_hessian + self.vdw_hessian + self.s14_elec_hessian + self.s14_vdw_hessian + self.dihedral_hessian + self.cmap_hessian


    def numeric_hessian(self, pos, delta=1e-3):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"

        self.bond_hessian_numeric = self.bonds.numeric_hessian(pos, delta)
        self.angle_hessian_numeric = self.angles.numeric_hessian(pos)
        self.dihedral_hessian_numeric, self.s14_elec_hessian_numeric, self.s14_vdw_hessian_numeric = self.dihedrals.numeric_hessian(pos,self.atoms,1e-2)
        self.elec_hessian_numeric, self.vdw_hessian_numeric = self.atoms.numeric_hessian(pos, delta)
        self.cmap_hessian_numeric = self.cmap.numeric_hessian(pos, delta)
        self.total_hessian_numeric = self.bond_hessian_numeric + self.angle_hessian_numeric + self.elec_hessian_numeric + self.vdw_hessian_numeric  + self.s14_elec_hessian_numeric + self.s14_vdw_hessian_numeric + self.dihedral_hessian_numeric + self.cmap_hessian_numeric
    
    def energy_minimize(self, pos, tol=1e-4, maxiter=10000):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"

        #self.min_obj = optimize.minimize(self._energy, pos.flatten(), method='Newton-CG', jac='3-point', hess='3-point')
        #self.min_obj = optimize.minimize(self._energy, pos.flatten(), method='BFGS', tol=tol, options = {'gtol': tol} )
        #self.min_obj = optimize.minimize(self._energy, pos.flatten(), method='Nelder-Mead', tol=tol, options = {'maxiter': maxiter})
        pos_flat = pos.flatten().astype(np.float64)
        self.min_obj = optimize.minimize(self._energy, pos.flatten(), method='CG', jac=self._gradient, tol=tol, options={'gtol': tol, 'disp': True, 'maxiter': maxiter})
        self.min_pos = self.min_obj.x.reshape(-1,3)
        self.min_energy = self.energy(self.min_pos)
        self.hessian(self.min_pos)
        #print("Minimized energy: ", self.min_energy)

    def _numeric_hessian(self, pos, delta=1.0e-3):
        
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"
        
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        delta2 = delta*delta
        # declare Hessian matrix
        hess = np.zeros((3*n_atoms,3*n_atoms),dtype=np.float64)
        # populate upper triangle
        for coord1 in range(3*n_atoms):
            coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
            coord1_delta[coord1] = delta
            #coord1_delta = coord1_delta.reshape(n_atoms,3)
            for coord2 in range(coord1,3*n_atoms):
                coord2_delta = np.zeros(3*n_atoms, dtype=np.float64)
                coord2_delta[coord2] = delta
                #coord2_delta = coord2_delta.reshape(n_atoms,3)
                hess[coord1,coord2] = hess[coord2,coord1] = ( self._energy(pos_flat+0.5*(coord1_delta+coord2_delta)) + self._energy(pos_flat-0.5*(coord1_delta+coord2_delta)) - self._energy(pos_flat+0.5*(coord1_delta-coord2_delta)) - self._energy(pos_flat-0.5*(coord1_delta-coord2_delta)) )
        # complete
        hess /= delta2
        return hess

    def _numeric_force(self, pos, delta=1.0e-3):
        # meta data
        n_atoms = pos.shape[0]
        pos_flat = pos.flatten().astype(np.float64)
        # declare force vector
        force = np.zeros(3*n_atoms,dtype=np.float64)
        coord1_delta = np.zeros(3*n_atoms,dtype=np.float64)
        # move each coordinate and calculated finite difference
        for coord1 in range(3*n_atoms):
            coord1_delta[coord1] = 0.5*delta
            force[coord1] = self._energy(pos_flat + coord1_delta) - self._energy(pos_flat - coord1_delta)
            coord1_delta[coord1] = 0.0
        # complete
        force /= delta
        return -force.reshape((n_atoms,3))
    
    def _energy(self, flattened_pos):
        pos = flattened_pos.reshape(-1,3)
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"
        
        energy = self.bonds.energy(pos)
        energy += self.angles.energy(pos)
        dihedral_energy, s14_elec, s14_vdw = self.dihedrals.energy(pos,self.atoms)
        elec_energy, vdw_energy = self.atoms.energy(pos)
        energy += self.cmap.energy(pos)
        energy += elec_energy + vdw_energy + s14_elec + s14_vdw + dihedral_energy
        #energy += elec_energy + vdw_energy
        return energy

    def _gradient(self, flattened_pos):

        pos = flattened_pos.reshape(-1,3)
        # Assert that position is of the right size
        assert pos.shape[0] == self.atoms.n_atoms, "postition array has incorrect number of atoms"
        
        bond_force = self.bonds.force(pos)
        angle_force = self.angles.force(pos)
        dihedral_force, s14_elec_force, s14_vdw_force = self.dihedrals.force(pos,self.atoms)
        elec_force, vdw_force = self.atoms.force(pos)
        cmap_force = self.cmap.force(pos)
        total_force = bond_force + angle_force + elec_force + vdw_force + s14_elec_force + s14_vdw_force + dihedral_force + cmap_force
        return -total_force.flatten()
    
    def _read_parm7(self):
        flagSearch = "FLAG"
        blank = " "
        metaDataFlag = "POINTERS"
        chargeFlag = "CHARGE"
        massFlag = "MASS"
        atomTypeIndexFlag = "ATOM_TYPE_INDEX"
        nExcludedAtomsFlag = "NUMBER_EXCLUDED_ATOMS"
        excludedAtomsListFlag = "EXCLUDED_ATOMS_LIST"
        nonBondedParmIndexFlag = "NONBONDED_PARM_INDEX"
        ljACoeffFlag = "LENNARD_JONES_ACOEF"
        ljBCoeffFlag = "LENNARD_JONES_BCOEF"
        bondKFlag = "BOND_FORCE_CONSTANT"
        bondX0Flag = "BOND_EQUIL_VALUE"
        bondnHFlag = "BONDS_WITHOUT_HYDROGEN"
        bondHFlag = "BONDS_INC_HYDROGEN"
        angleKFlag = "ANGLE_FORCE_CONSTANT"
        angleX0Flag = "ANGLE_EQUIL_VALUE"
        anglenHFlag = "ANGLES_WITHOUT_HYDROGEN"
        angleHFlag = "ANGLES_INC_HYDROGEN"
        dihKFlag = "DIHEDRAL_FORCE_CONSTANT"
        dihNFlag = "DIHEDRAL_PERIODICITY"
        dihPFlag = "DIHEDRAL_PHASE"
        dihnHFlag = "DIHEDRALS_WITHOUT_HYDROGEN"
        dihHFlag = "DIHEDRALS_INC_HYDROGEN"
        sceeScaleFactorFlag = "SCEE_SCALE_FACTOR"
        scnbScaleFactorFlag = "SCNB_SCALE_FACTOR"
        atomsPerMoleculeFlag = "ATOMS_PER_MOLECULE"
        solventPointerFlag = "SOLVENT_POINTERS"
        cmapCountFlag = "CMAP_COUNT"
        cmapResolutionFlag = "CMAP_RESOLUTION"
        cmapParameterFlag = "CMAP_PARAMETER"
        cmapIndexFlag = "CMAP_INDEX"
        
        prmFile = open(self.parm7_file_name, "r")
        
        for line in prmFile:
            if flagSearch in line:
                flag = line.split(' ')[1]
                if metaDataFlag == flag:
                    #read meta data
                    print("Reading system metadata from prmtop file")
        			#skip format line */
                    temp = prmFile.readline()
        			# read meta data section line by line */
        			# line 1: */
                    temp = prmFile.readline()
                    n_atoms = int(temp[:8])
                    #
                    print("Number of atoms from prmtop file: %d" % n_atoms)
                    n_types = int(temp[8:16])
                    print("Number of atom types from prmtop file: %d" % n_types)
                    n_bonds_H = int(temp[16:24]) 
                    print("Number of bonds containing hydrogens: %d" % n_bonds_H)
                    n_bonds_noH = int(temp[24:32])
                    print("Number of bonds NOT containing hydrogens: %d" % n_bonds_noH)
                    n_bonds = n_bonds_H + n_bonds_noH
                    n_angles_H = int(temp[32:40])
                    print("Number of angles containing hydrogens: %d" % n_angles_H);
                    n_angles_noH = int(temp[40:48])
                    print("Number of angles NOT containing hydrogens: %d" % n_angles_noH)
                    n_dihedrals_H = int(temp[48:56])
                    print("Number of dihs containing hydrogens: %d" % n_dihedrals_H)
                    n_dihedrals_noH = int(temp[56:64])
                    print("Number of dihs NOT containing hydrogens: %d" % n_dihedrals_noH)
        			# line 2: */
                    temp = prmFile.readline()
                    excluded_atoms_list_length = int(temp[:8])
                    print("Length of excluded atoms list: %d" % excluded_atoms_list_length)
                    bond_n_types = int(temp[40:48])
                    print("Number of unique bond types: %d" % bond_n_types)
                    angle_n_types = int(temp[48:56])
                    print("Number of unique angle types: %d" % angle_n_types)
                    dihedral_n_types = int(temp[56:64])
                    print("Number of unique dih types: %d" % dihedral_n_types)
                    # declare atom, bond, angle, and dihedral objects
                    self.atoms = atoms(n_atoms, n_types, excluded_atoms_list_length)
                    self.bonds = bonds(n_bonds_H, n_bonds_noH, bond_n_types)
                    self.angles = angles(n_angles_H, n_angles_noH, angle_n_types)
                    self.dihedrals = dihedrals(n_dihedrals_H, n_dihedrals_noH, dihedral_n_types)
                elif chargeFlag in flag:
        			# read bond k values
                    nLines = int( (n_atoms + 4) / 5.0 )
        			# skip format line */
                    temp = prmFile.readline()
        			# loop over lines */
                    atomCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (atomCount < n_atoms and lineCount < 5):
                            # charge will be fourth element in position float4 array
                            self.atoms.charges[atomCount] = float( temp[lineCount*16:lineCount*16+16] )  #/sqrtEps;
                            atomCount += 1
                            lineCount += 1
                elif massFlag in flag:
					# read bond k values
                    nLines = int( (n_atoms + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					#loop over lines */
                    atomCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (atomCount < n_atoms and lineCount < 5):
                            self.atoms.masses[atomCount] = float(temp[lineCount*16:lineCount*16+16])
                            atomCount += 1
                            lineCount += 1
                elif atomTypeIndexFlag in flag:
					# 
                    nLines = int( (n_atoms + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    atomCount = 0;
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (atomCount < n_atoms and lineCount < 10):
                            self.atoms.ityp[atomCount] = int(temp[lineCount*8:lineCount*8+8])-1 # minus one for C zero indexing
                            atomCount += 1
                            lineCount += 1
                    
                elif nExcludedAtomsFlag in flag:
					# 
                    nLines = int( (n_atoms + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    atomCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (atomCount < n_atoms and lineCount < 10):
                            self.atoms.n_excluded_atoms[atomCount] = int(temp[lineCount*8:lineCount*8+8])
                            atomCount += 1
                            lineCount += 1
                elif nonBondedParmIndexFlag in flag:
					# 
                    n_types_2 = n_types*n_types;
                    nLines = int( (n_types_2 + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    parmCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (parmCount < n_types_2 and lineCount < 10):
                            self.atoms.nonbonded_parm_index[parmCount] = int(temp[lineCount*8:lineCount*8+8])-1 # subtract one here since C is zero indexed
                            parmCount += 1
                            lineCount += 1
                elif bondX0Flag in flag:
					#read bond k values
                    nLines = int ((self.bonds.n_types + 4) / 5.0 )
					#skip format line */
                    temp = prmFile.readline()
					#loop over lines */
                    bondCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (bondCount < self.bonds.n_types and lineCount < 5):
                            self.bonds.bond_params[bondCount,1] = float(temp[lineCount*16:lineCount*16+16])
                            bondCount += 1
                            lineCount += 1
                elif bondKFlag in flag:
					#read bond k values
                    nLines = int ((self.bonds.n_types + 4) / 5.0 )
					#skip format line */
                    temp = prmFile.readline()
					#
                    bondCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (bondCount < self.bonds.n_types and lineCount < 5):
                            self.bonds.bond_params[bondCount,0] = float(temp[lineCount*16:lineCount*16+16]) # multiply by two ?
                            bondCount += 1
                            lineCount += 1
                elif angleX0Flag in flag:
					# read angle k values
                    nLines = int( (self.angles.n_types + 4) / 5.0)
					# skip format line */
                    temp = prmFile.readline()
					#loop over lines */
                    angleCount = 0;
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (angleCount < self.angles.n_types and lineCount < 5):
                            self.angles.angle_params[angleCount,1] = float(temp[lineCount*16:lineCount*16+16])
                            angleCount += 1
                            lineCount += 1
                elif angleKFlag in flag:
					#read angle k values
                    nLines = int((self.angles.n_types + 4) / 5.0 )
					#skip format line */
                    temp = prmFile.readline()
					#loop over lines */
                    angleCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (angleCount < self.angles.n_types and lineCount < 5):
                            self.angles.angle_params[angleCount,0] = float(temp[lineCount*16:lineCount*16+16]) # should I multiply by 2?
                            angleCount += 1
                            lineCount += 1
                elif dihNFlag in flag:
					# read dih periodicity values
                    nLines = int( (self.dihedrals.n_types + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_types and lineCount < 5) :
                            self.dihedrals.dihedral_params[dihCount,1] = float(temp[lineCount*16:lineCount*16+16])
                            dihCount += 1
                            lineCount += 1
                elif dihKFlag in flag:
					# read dih k values
                    nLines = int ( (self.dihedrals.n_types + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_types and lineCount < 5):
                            self.dihedrals.dihedral_params[dihCount,0] = float(temp[lineCount*16:lineCount*16+16])
                            dihCount += 1
                            lineCount += 1
                elif dihPFlag in flag:
					# read dih phase values
                    nLines = int ( (self.dihedrals.n_types + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_types and lineCount < 5):
                            self.dihedrals.dihedral_params[dihCount,2] = float(temp[lineCount*16:lineCount*16+16])
                            dihCount += 1
                            lineCount += 1
                elif sceeScaleFactorFlag in flag:
					# read dih phase values
                    nLines = int ( (self.dihedrals.n_types + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_types and lineCount < 5):
                            self.dihedrals.scaled_14_factors[dihCount,0] = float(temp[lineCount*16:lineCount*16+16])
                            if self.dihedrals.scaled_14_factors[dihCount,0] > 0:
                                self.dihedrals.scaled_14_factors[dihCount,0] = 1.0 / self.dihedrals.scaled_14_factors[dihCount,0]
                            dihCount += 1
                            lineCount += 1
                elif scnbScaleFactorFlag in flag:
					# read dih phase values
                    nLines = int ( (self.dihedrals.n_types + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_types and lineCount < 5):
                            self.dihedrals.scaled_14_factors[dihCount,1] = float(temp[lineCount*16:lineCount*16+16])
                            if self.dihedrals.scaled_14_factors[dihCount,1] > 0:
                                self.dihedrals.scaled_14_factors[dihCount,1] = 1.0 / self.dihedrals.scaled_14_factors[dihCount,1]
                            dihCount += 1
                            lineCount += 1
                elif ljACoeffFlag in flag:
                    nTypes2 = self.atoms.n_types*(self.atoms.n_types+1)//2
                    nLines = int ( (nTypes2 + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    typeCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (typeCount < nTypes2 and lineCount < 5):
                            self.atoms.lj_params[typeCount,0] = float(temp[lineCount*16:lineCount*16+16])
                            typeCount += 1
                            lineCount += 1
                elif ljBCoeffFlag in flag:
                    nTypes2 = self.atoms.n_types*(self.atoms.n_types+1)//2
                    nLines = int ( (nTypes2 + 4) / 5.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    typeCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (typeCount < nTypes2 and lineCount < 5):
                            self.atoms.lj_params[typeCount,1] = float(temp[lineCount*16:lineCount*16+16])
                            typeCount += 1
                            lineCount += 1
                elif bondHFlag in flag:
					# FORMAT 10I8 */
                    nLines = int ( (self.bonds.n_bonds_H*3 + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    bondCount = 0
                    tempBondArray = np.empty(self.bonds.n_bonds_H*3)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (bondCount < self.bonds.n_bonds_H*3 and lineCount < 10) :
                            tempBondArray[bondCount] = int(temp[lineCount*8:lineCount*8+8])
                            bondCount += 1
                            lineCount += 1
                	# parse to bond arrays
                    for i in range(self.bonds.n_bonds_H) :
                        self.bonds.bond_atoms[i,0] = tempBondArray[i*3]//3      # first bond atom
                        self.bonds.bond_atoms[i,1] = tempBondArray[i*3+1]//3    # second bond atom
                        self.bonds.bond_atoms[i,2] = tempBondArray[i*3+2]-1     # bond type
                elif bondnHFlag in flag:
					# FORMAT 10I8 */
                    nLines = int ( (self.bonds.n_bonds_noH*3 + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    bondCount = 0
                    tempBondArray = np.empty(self.bonds.n_bonds_noH*3)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (bondCount < self.bonds.n_bonds_noH*3 and lineCount < 10) :
                            tempBondArray[bondCount] = int(temp[lineCount*8:lineCount*8+8])
                            bondCount += 1
                            lineCount += 1
                	# parse to bond arrays
                    for i in range(self.bonds.n_bonds_H,self.bonds.n_bonds) :
                        self.bonds.bond_atoms[i,0] = tempBondArray[(i-self.bonds.n_bonds_H)*3]//3      # first bond atom
                        self.bonds.bond_atoms[i,1] = tempBondArray[(i-self.bonds.n_bonds_H)*3+1]//3    # second bond atom
                        self.bonds.bond_atoms[i,2] = tempBondArray[(i-self.bonds.n_bonds_H)*3+2]-1     # bond type
                elif angleHFlag in flag: 
					# FORMAT 10I8 */
                    nLines = int ((self.angles.n_angles_H*4 + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    angleCount = 0
                    tempAngleArray = np.empty(self.angles.n_angles_H*4)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (angleCount < self.angles.n_angles_H*4 and lineCount < 10):
                            tempAngleArray[angleCount] = int(temp[lineCount*8:lineCount*8+8])
                            angleCount += 1
                            lineCount +=1
					# parse to angle arrays
                    for i in range(self.angles.n_angles_H):
                        self.angles.angle_atoms[i,0] = tempAngleArray[i*4]//3
                        self.angles.angle_atoms[i,1] = tempAngleArray[i*4+1]//3
                        self.angles.angle_atoms[i,2] = tempAngleArray[i*4+2]//3
                        self.angles.angle_atoms[i,3] = tempAngleArray[i*4+3]-1
                elif anglenHFlag in flag: 
					# FORMAT 10I8 */
                    nLines = int ((self.angles.n_angles_noH*4 + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    angleCount = 0
                    tempAngleArray = np.empty(self.angles.n_angles_noH*4)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (angleCount < self.angles.n_angles_noH*4 and lineCount < 10):
                            tempAngleArray[angleCount] = int(temp[lineCount*8:lineCount*8+8])
                            angleCount += 1
                            lineCount +=1
					# parse to angle arrays
                    for i in range(self.angles.n_angles_H, self.angles.n_angles):
                        self.angles.angle_atoms[i,0] = tempAngleArray[(i-self.angles.n_angles_H)*4]//3
                        self.angles.angle_atoms[i,1] = tempAngleArray[(i-self.angles.n_angles_H)*4+1]//3
                        self.angles.angle_atoms[i,2] = tempAngleArray[(i-self.angles.n_angles_H)*4+2]//3
                        self.angles.angle_atoms[i,3] = tempAngleArray[(i-self.angles.n_angles_H)*4+3]-1   # angle type
                elif dihHFlag in flag:
					# FORMAT 10I8 */
                    nLines = int ( (self.dihedrals.n_dihedrals_H*5 + 9) / 10.0 )
                    # skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    tempDihArray = np.empty(self.dihedrals.n_dihedrals_H*5)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_dihedrals_H*5 and lineCount < 10):
                            tempDihArray[dihCount] = int(temp[lineCount*8:lineCount*8+8]) 
                            dihCount += 1
                            lineCount += 1
					# parse to dih arrays
                    for i in range(self.dihedrals.n_dihedrals_H):
                        self.dihedrals.dihedral_atoms[i,0] = tempDihArray[i*5]//3
                        self.dihedrals.dihedral_atoms[i,1] = tempDihArray[i*5+1]//3
                        self.dihedrals.dihedral_atoms[i,2] = tempDihArray[i*5+2]//3
                        self.dihedrals.dihedral_atoms[i,3] = tempDihArray[i*5+3]//3
                        self.dihedrals.dihedral_atoms[i,4] = tempDihArray[i*5+4]-1    # dihedral type
                elif dihnHFlag in flag:
					# FORMAT 10I8 */
                    nLines = int ( (self.dihedrals.n_dihedrals_noH*5 + 9) / 10.0 )
                    # skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    dihCount = 0
                    tempDihArray = np.empty(self.dihedrals.n_dihedrals_noH*5)
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (dihCount < self.dihedrals.n_dihedrals_noH*5 and lineCount < 10):
                            tempDihArray[dihCount] = int(temp[lineCount*8:lineCount*8+8]) 
                            dihCount += 1
                            lineCount += 1
					# parse to dih arrays
                    for i in range(self.dihedrals.n_dihedrals_H,self.dihedrals.n_dihedrals):
                        self.dihedrals.dihedral_atoms[i,0] = tempDihArray[(i-self.dihedrals.n_dihedrals_H)*5]//3
                        self.dihedrals.dihedral_atoms[i,1] = tempDihArray[(i-self.dihedrals.n_dihedrals_H)*5+1]//3
                        self.dihedrals.dihedral_atoms[i,2] = tempDihArray[(i-self.dihedrals.n_dihedrals_H)*5+2]//3
                        self.dihedrals.dihedral_atoms[i,3] = tempDihArray[(i-self.dihedrals.n_dihedrals_H)*5+3]//3
                        self.dihedrals.dihedral_atoms[i,4] = tempDihArray[(i-self.dihedrals.n_dihedrals_H)*5+4]-1    # dihedral type
                elif excludedAtomsListFlag in flag:
					# 
                    nLines = int( (self.atoms.excluded_atoms_list_length + 9) / 10.0 )
					# skip format line */
                    temp = prmFile.readline()
					# loop over lines */
                    atomCount = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (atomCount < self.atoms.excluded_atoms_list_length and lineCount < 10):
                            self.atoms.excluded_atoms_list[atomCount] = int(temp[lineCount*8:lineCount*8+8])-1
                            atomCount += 1
                            lineCount += 1
                elif cmapCountFlag in flag:
                    # skip format line
                    temp = prmFile.readline()
                    # read data line
                    temp = prmFile.readline()
                    cmap_term_count = int(temp[:8])
                    cmap_type_count = int(temp[8:16])
                    self.cmap = cmap(cmap_term_count, cmap_type_count)
                elif cmapResolutionFlag in flag:
                    nLines = int( (self.cmap.term_count + 19) / 20.0 )
                    # skip format line
                    temp = prmFile.readline()
                    # read data lines
                    cmap_term_count = 0
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (cmap_term_count < self.cmap.term_count and lineCount < 20):
                            self.cmap.resolution[cmap_term_count] = int(temp[lineCount*4:lineCount*4+4])
                            lineCount += 1
                            cmap_term_count += 1
                    # 
                    self.cmap.allocate_parameter_list()
                elif cmapParameterFlag in flag:
                    # skip two format lines
                    temp = prmFile.readline()
                    temp = prmFile.readline()
                    # 
                    cmap_parameter_count = 0
                    cmap_parameter_limit = self.cmap.resolution[self.cmap.parameter_count]**2
                    nLines = int( (cmap_parameter_limit + 7) / 8.0 )
                    for i in range(nLines):
                        temp = prmFile.readline()
                        lineCount = 0
                        while (cmap_parameter_count < cmap_parameter_limit and lineCount < 8):
                            self.cmap.parameter_list[self.cmap.parameter_count][cmap_parameter_count // self.cmap.resolution[self.cmap.parameter_count], cmap_parameter_count % self.cmap.resolution[self.cmap.parameter_count]] = float(temp[lineCount*9:lineCount*9+9])
                            lineCount += 1
                            cmap_parameter_count += 1
                    #
                    self.cmap.parameter_count += 1
                elif cmapIndexFlag in flag:
                    # skip format line
                    temp = prmFile.readline()
                    #
                    nLines = self.cmap.term_count
                    for i in range(nLines):
                        temp = prmFile.readline()
                        for j in range(6):
                            self.cmap.atom_list[i,j] = int(temp[j*8:j*8+8])-1
        self.cmap.fit_interpolators()
        prmFile.close()
