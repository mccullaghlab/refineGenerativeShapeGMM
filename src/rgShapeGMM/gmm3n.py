from shapeGMMTorch import align 
import torch
import numpy as np
from scipy import stats

def gen_traj(num, covar, center, dtype=torch.float64, device=torch.device("cpu")):
    """
    This function generates samples for individual cluster.
    
    num        - (integer) number of samples to be generated from the cluster
    covar      - (3Nx3N array) sample covaraince of the cluster 
    center     - (Nx3 array) center of the cluster
    
    """
    # meta data
    n_atoms = center.shape[0]
    # computes the eigen value decomposition of covariance 
    e, v = torch.linalg.eigh(covar)
    
    # setting first 6 eigen values to zero 
    # (null spaces corresponding to 3 trans and 3 rots dofs for  3Nx3N sample covariance)
    e[:6] = 0.0
    # calculate std dev for each normal mode
    e[6:] = torch.sqrt(e[6:])
    
    # generate normally distributed random variables (mean 0, stdev 1)
    norms = torch.normal(0, 1, size=(covar.shape[0],num), dtype=dtype, device=device)
    
    # multiply by normal mode stdev
    norms *= e.view((-1,1))

    # rotate back into original basis
    trj = torch.matmul(v,norms)
    trj = trj.T.reshape(num,n_atoms,3)
    trj += center 
    
    return trj
    
def cluster_ids_from_rand(random_nums, weights):
    """
    This function randomly generates cluster ids cosindering weights of different clusters.
    """
    running_sum = 0.0
    cluster_ids = np.empty(random_nums.size,dtype=np.int32)
    for cluster_count, weight in enumerate(weights):
        cluster_ids[np.argwhere((random_nums > running_sum) &
                            (random_nums < running_sum + weight)).flatten()] = cluster_count
        running_sum += weight
    return cluster_ids 
        
def kl_divergence(sgmmP, sgmmQ, n_points):
    """
    Compute the Kullback-Leibler divergence, Dkl(P||Q), from sgmmQ (Q) to sgmmP (P) by sampling from sgmmP
    with n_points

    sgmmP             : reference shapeGMM object
    sgmmQ             : target shapeGMM object
    n_points          : (integer) number of frames to generate to estimate KL divergence

    returns:
    lnP - lnQ         : (float) KL divergence
    sterr(lnP - lnQ)  : (float) standard error of sampled KL divergence
    """
    trj = sgmmP.generate(n_points)
    lnP = sgmmP.predict(trj)[1]  # LL per frame 
    lnQ = sgmmQ.predict(trj)[1]  # LL per frame
    return lnP - lnQ, stats.sem(sgmmP.predict_frame_log_likelihood-sgmmQ.predict_frame_log_likelihood)

class gmm3N:
    """
    class to create a GMM with 3Nx3N covariance from means, covars and weights
    """
    
    def __init__(self, centers, covars, weights, device = torch.device("cpu"), dtype = torch.float64):
        """
        
        Initialize instance varibales.
        
        centers             - Input trajectory frames (n_frames x n_atoms x 3) 
        covars              - Input array of cluster covariances
        weights             - Input array of cluster weights
        
        """
        # assign self variables
        self.device = device
        self.dtype = dtype
        self.n_atoms = centers.shape[1]
        self.n_clusters  = centers.shape[0]
        self.weights = weights
        self.centers =  centers
        self.covars = covars
  
    # predict clustering of provided data based on prefit parameters from fit_weighted
    def predict(self,traj_data):
        """
        Predict size-and-shape GMM using traj_data as prediction set and already fit object parameters.
        traj_data (required)     - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 

        Returns:
        cluster ids             - (n_frames) int array
        log likelihood          - float64 scalar of the log likelihood of the data given the fit model
        """

        # metadata
        n_frames = traj_data.shape[0]
        rank = self.n_atoms*3-6
        # center trajectory
        traj_data -= np.mean(traj_data,axis=1,keepdims=True)
        # send data to device
        traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
        centers_tensor = torch.tensor(self.centers,dtype=self.dtype,device=self.device)
        ln_weights_tensor = torch.tensor(np.log(self.weights),dtype=torch.float64,device=self.device)
        # declare covariance matrices 
        covars_tensor = torch.tensor(self.covars,dtype=torch.float64,device=self.device)
        # declare array to populate
        cluster_frame_ln_likelihoods_tensor = torch.empty((n_frames, self.n_clusters),dtype=torch.float64, device=self.device)
        for cluster in range(self.n_clusters):
            # create precision
            e, v = torch.linalg.eigh(covars_tensor[cluster])
            lpdet = torch.sum(torch.log(e[6:]))
            e[:6] = 0.0
            e[6:] = 1/e[6:]
            precision = v @ torch.diag(e) @ v.T
            # uniform alignment of trajectory
            aligned_traj_tensor = align.align_uniform(traj_tensor,centers_tensor[cluster])
            # compute displacement
            disp = (aligned_traj_tensor - centers_tensor[cluster]).reshape((-1,self.n_atoms*3))
            # compute LL
            quad = disp.view(-1,1,self.n_atoms*3) @ precision @ disp.view(-1,self.n_atoms*3,1)
            cluster_frame_ln_likelihoods_tensor[:,cluster] = -0.5 * quad.view(-1) - 0.5*rank*torch.log(torch.tensor(2*np.pi,device=self.device)) - 0.5*lpdet + ln_weights_tensor[cluster]
        # compute LL per frame
        self.predict_frame_log_likelihood = (torch.logsumexp(cluster_frame_ln_likelihoods_tensor,1)).cpu().numpy()
        log_likelihood = np.mean(self.predict_frame_log_likelihood)
        # assign clusters based on largest likelihood (probability density)
        clusters = torch.argmax(cluster_frame_ln_likelihoods_tensor, dim = 1).cpu().numpy()
        # delete data from gpu
        del traj_tensor
        del ln_weights_tensor
        del centers_tensor
        del covars_tensor
        del precision
        del lpdet
        torch.cuda.empty_cache()
        # return values
        return clusters, log_likelihood
    
    def generate(self, num_samples=10000):
        """
        This function generates frames from all the clusters.
        """
        
        cluster_ids = cluster_ids_from_rand(np.random.rand(num_samples), self.weights)
        traj_out = np.empty((num_samples, self.n_atoms, 3))
        
        for cluster_id in range(self.n_clusters):
            
            indeces = np.argwhere(cluster_ids == cluster_id).flatten()
            
            # generate samples for this particular cluster
            traj_out[indeces] = gen_traj(indeces.size, torch.tensor(self.covars[cluster_id],device=self.device), torch.tensor(self.centers[cluster_id],device=self.device)).cpu().numpy()
        
        return traj_out

    def configurational_entropy(self, n_points=10000):
        """
        Compute the configurational entropy of this object in units of R
        with n_points

        n_points   : (integer) number of frames to generate to sample

        returns:
        -lnP       : (float) configurational entropy in units of R
        sterr(lnP) : (float) standard error of sampled configurational entropy in units of R
        """
        # sample the object and compute probabilities of each sampled point
        trj = self.generate(n_points)
        lnP = self.predict(trj)[1]
        return -lnP + 0.5*(3*(self.n_atoms-2))*np.log(2*np.pi), stats.sem(self.predict_frame_log_likelihood)
