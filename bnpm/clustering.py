import sklearn
import numpy as np
import matplotlib.pyplot as plt

import torch

import copy

from . import torch_helpers, indexing


def cluster_similarity_matrices(
    s, 
    l, 
    verbose=True
):
    """
    Compute the similarity matrices for each cluster in l.
    This algorithm works best on large and sparse matrices. 
    RH 2023

    Args:
        s (scipy.sparse.csr_matrix or np.ndarray or sparse.COO):
            Similarity matrix.
            Entries should be non-negative floats.
        l (np.ndarray):
            Labels for each row of s.
            Labels should be integers ideally.
        verbose (bool):
            Whether to print warnings.

    Returns:
        cs_mean (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the mean similarity between all the pairs
             of samples in each cluster.
            Note that the diagonal here only considers non-self similarity,
             which excludes the diagonals of s.
        cs_max (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the maximum similarity between all the pairs
             of samples in each cluster.
            Note that the diagonal here only considers non-self similarity,
             which excludes the diagonals of s.
        cs_min (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the minimum similarity between all the pairs
             of samples in each cluster. Will be 0 if there are any sparse
             elements between the two clusters.
    """
    import sparse
    import scipy.sparse

    l_arr = np.array(l)
    ss = scipy.sparse.csr_matrix(s.astype(np.float32))

    ## assert that all labels have at least two samples
    l_u ,l_c = np.unique(l_arr, return_counts=True)
    # assert np.all(l_c >= 2), "All labels must have at least two samples."
    ## assert that s is a square matrix
    assert ss.shape[0] == ss.shape[1], "Similarity matrix must be square."
    ## assert that s is non-negative
    assert (ss < 0).sum() == 0, "Similarity matrix must be non-negative."
    ## assert that l is a 1-D array
    assert len(l.shape) == 1, "Labels must be a 1-D array."
    ## assert that l is the same length as s
    assert len(l) == ss.shape[0], "Labels must be the same length as the similarity matrix."
    if verbose:
        ## Warn if s is not symmetric
        if not (ss - ss.T).sum() == 0:
            print("Warning: Similarity matrix is not symmetric.") if verbose else None
        ## Warn if s is not sparse
        if not isinstance(ss, (np.ndarray, sparse.COO, scipy.sparse.csr_matrix)):
            print("Warning: Similarity matrix is not a recognized sparse type or np.ndarray. Will attempt to convert to sparse.COO") if verbose else None
        ## Warn if diagonal is not all ones. It will be converted
        if not np.allclose(np.array(ss[range(ss.shape[0]), range(ss.shape[0])]), 1):
            print("Warning: Similarity matrix diagonal is not all ones. Will set diagonal to all ones.") if verbose else None
        ## Warn if there are any values greater than 1
        if (ss > 1).sum() > 0:
            print("Warning: Similarity matrix has values greater than 1.") if verbose else None
        ## Warn if there are NaNs. Set to 0.
        if (np.isnan(ss.data)).sum() > 0:
            print("Warning: Similarity matrix has NaNs. Will set to 0.") if verbose else None
            ss.data[np.isnan(ss.data)] = 0

    ## Make a boolean matrix for labels
    l_bool = sparse.COO(np.stack([l_arr == u for u in l_u], axis=0))
    samp_per_clust = l_bool.sum(1).todense()
    n_clusters = len(samp_per_clust)
    n_samples = ss.shape[0]
    
    ## Force diagonal to be 1s
    ss = ss.tolil()
    ss[range(n_samples), range(n_samples)] = 1
    ss = sparse.COO(ss)

    ## Compute the similarity matrix for each pair of clusters
    s_big_conj = ss[None,None,:,:] * l_bool[None,:,:,None] * l_bool[:,None,None,:]  ## shape: (n_clusters, n_clusters, n_samples, n_samples)
    s_big_diag = sparse.eye(n_samples) * l_bool[None,:,:,None] * l_bool[:,None,None,:]

    ## Compute the mean similarity matrix for each cluster
    samp_per_clust_crossGrid = samp_per_clust[:,None] * samp_per_clust[None,:]  ## shape: (n_clusters, n_clusters). This is the product of the number of samples in each cluster. Will be used to divide by the sum of similarities.
    norm_mat = samp_per_clust_crossGrid.copy()  ## above variable will be used again and this one will be mutated.
    fixed_diag = samp_per_clust * (samp_per_clust - 1)  ## shape: (n_clusters,). For the diagonal, we need to subtract 1 from the number of samples in each cluster because samples have only 1 similarity with themselves along the diagonal.
    norm_mat[range(n_clusters), range(n_clusters)] = fixed_diag  ## Correcting the diagonal
    s_big_sum_raw = s_big_conj.sum(axis=(2,3)).todense()
    s_big_sum_raw[range(n_clusters), range(n_clusters)] = s_big_sum_raw[range(n_clusters), range(n_clusters)] - samp_per_clust  ## subtract off the number of samples in each cluster from the diagonal
    cs_mean = s_big_sum_raw / norm_mat  ## shape: (n_clusters, n_clusters). Compute mean by finding the sum of the similarities and dividing by the norm_mat.

    ## Compute the min similarity matrix for each cluster
    ### This is done in two steps:
    #### 1. Compute the minimum similarity between each pair of clusters by inverting the similarity matrix and finding the maximum similarity between each pair of clusters.
    #### 2. Since the first step doesn't invert any values that happen to be 0 (since they are sparse), we need to find out if there are any 0 values there are in each cluster pair, and if there then the minimum similarity between the two clusters is 0.
    val_max = s_big_conj.max() + 1
    cs_min = s_big_conj.copy()
    cs_min.data = val_max - cs_min.data  ## Invert the values
    cs_min = cs_min.max(axis=(2,3))  ## Find the max similarity
    cs_min.data = val_max - cs_min.data  ## Invert the values back
    cs_min.fill_value = 0.0  ## Set the fill value to 0.0 since it gets messed up by these subtraction operations
    
    n_missing_values = (samp_per_clust_crossGrid - (s_big_conj > 0).sum(axis=(2,3)).todense())  ## shape: (n_clusters, n_clusters). Compute the number of missing values by subtracting the number of non-zero values from the number of samples in each cluster.
    # n_missing_values[range(len(samp_per_clust)), range(len(samp_per_clust))] = (samp_per_clust**2 - samp_per_clust) - ((s_big_conj[range(len(samp_per_clust)), range(len(samp_per_clust))] > 0).sum(axis=(1,2))).todense()  ## Correct the diagonal by subtracting the number of non-zero values from the number of samples in each cluster. This is because the diagonal is the number of samples in each cluster squared minus the number of samples in each cluster.
    bool_nonMissing_values = (n_missing_values == 0)  ## shape: (n_clusters, n_clusters). Make a boolean matrix for where there are no missing values.
    cs_min = cs_min.todense() * bool_nonMissing_values  ## Set the minimum similarity to 0 where there are missing values.

    ## Compute the max similarity matrix for each cluster
    cs_max = (s_big_conj - s_big_diag).max(axis=(2,3))

    return cs_mean, cs_max.todense(), cs_min


class cDBSCAN():
    """
    Consensus DBSCAN algorithm.
    Runs DBSCAN on each epsilon value and returns the unique
    clusters as well as the number of times each cluster was
    found across epsilon values.
    RH 2022
    
    Args:
        X (np.ndarray):
            The data matrix.
        eps_values (list of float):
            The epsilon values to use in DBSCAN.
        min_samples (int):
            The minimum number of samples to use in DBSCAN.
        **kwargs_dbscan (dict):
            The optional arguments to pass to DBSCAN.
            Includes:
                metric_params (dict): default=None
                    The metric parameters to pass to DBSCAN.
                algorithm (str): default='auto'
                    The algorithm to use in DBSCAN.
                leaf_size (int): default=30
                    The leaf size to use in DBSCAN.
                p (int): default=2
                    The p value to use in DBSCAN.
                n_jobs (int): default=-1
                    The number of jobs to use in DBSCAN.   
    Returns:
        clusters_idx_unique (list of np.ndarray):
            The sample indices for each unique cluster
        clusters_idx_unique_freq (list of int):
            The number of times each unique cluster was found
            over the different epsilon values.
    """
    def __init__(
        self,
        eps_values=[0.1, 0.5, 2.5],
        min_samples=2,
        **kwargs_dbscan
    ):

        # eps_values = np.arange(1.0, 5.0, 0.1)
        # X = block['embeddings']
        # min_samples = 2,

        # kwargs_dbscan = {
        #     'metric_params': None, 
        #     'algorithm': 'auto',
        #     'leaf_size': 30, 
        #     'p': 2, 
        #     'n_jobs': -1
        # }

        self.dbscan_objs = [sklearn.cluster.DBSCAN(
                eps=eps,
                min_samples=min_samples, 
                **kwargs_dbscan
            ) for eps in eps_values]

    def _labels_to_idx(self, labels):
    #     return {label: np.where(labels==label)[0] for label in np.unique(labels)}
        return [np.where(labels==label)[0] for label in np.unique(labels)]

    def _freq_of_values(self, vals):
        u = np.unique(vals)
        f = np.array([np.sum(vals==unique) for unique in u])
        return np.array([f[u==val][0] for val in vals])
            

    def fit(self, X):
        # cDBSCAN: 'consensus DBSCAN': runs a DBSCAN over a sweep of epsilon values and records how many times each cluster appeared
        cluster_idx = []
        for ii, dbscan in enumerate(self.dbscan_objs):
            ## DBSCAN
            db = copy.deepcopy(dbscan)
            db.fit(X) # note that db.labels_==-1 means no cluster found
            db.labels_[self._freq_of_values(db.labels_) < dbscan.min_samples] = -1 # fail safe because sometimes there are clusters of just 1 for some reason...

            # concatenate the indices of clusters found at this eps value to a big list
            [cluster_idx.append(idx) for idx in self._labels_to_idx(db.labels_)]

        clusterHashes_block = [hash(tuple(vec)) for vec in cluster_idx]
        u, idx, c = np.unique(
            ar=clusterHashes_block,
            return_index=True,
            return_counts=True,
        )
        self.clusters_idx_unique = np.array(cluster_idx, dtype=object)[idx]
        self.clusters_idx_unique_freq = c

        return self.clusters_idx_unique, self.clusters_idx_unique_freq


# class Constrained_rich_clustering:
#     """
#     Class to perform constrained cluster assignment.
#     This method takes in putative clusters, a cluster similarity matrix,
#      and a vector of cluster 'scores' describing how valuable each
#      cluster is. It then attempts to find an optimal combination of
#      clusters. This is done by minimizing inclusion of similar clusters
#      and maximizing inclusion of clusters with high scores.
#     The cluster similarity matrix and score vector can be made using 
#      the cluster_dispersion_score and cluster_silhouette_score functions
#      respectively. The cluster membership matrix showing putative clusters
#      can be made by by doing something like sweeping over linkage distances.

#     RH 2022
#     """
#     def __init__(
#         self,
#         c,
#         h,
#         w=None,
#         m_init=None,
#         device='cpu',
#         optimizer_partial=None,
#         scheduler_partial=None,
#         dmCEL_temp=1,
#         dmCEL_sigSlope=2,
#         dmCEL_sigCenter=0.5,
#         dmCEL_penalty=1,
#         sampleWeight_softplusKwargs={'beta': 500, 'threshold': 50},
#         sampleWeight_penalty=1e1,
#         fracWeighted_goalFrac=1.0,
#         fracWeighted_sigSlope=2,
#         fracWeighted_sigCenter=0.5,
#         fracWeight_penalty=1e0,
#         maskL1_penalty=0e-3,
#         tol_convergence=1e-2,
#         window_convergence=100,
#         freqCheck_convergence=100,
#         verbose=True,
#     ):
#         """
#         Initialize the cluster assignment class.

#         Args:
#             c (scipy.sparse.csr_matrix, dtype float):
#                 'Cluster similarity' matrix.
#                 Elements are pairwise similarities between clusters.
#                 shape: (n_clusters, n_clusters)
#             h (scipy.sparse.csr_matrix, dtype bool):
#                 'multi Hot' boolean matrix.
#                 The cluster membership matrix.
#                 shape: (n_samples, n_clusters)
#             w (torch.Tensor, dtype float):
#                 'Weight' vector.
#                 Elements are the scores of each cluster. Can be customized.
#                 Weighs how much to value the inclusion of each cluster
#                  relative to the total fracWeight_penalty.
#                 shape: (n_clusters)
#                 If None, the clusters are weighted equally.
#             m_init (torch.Tensor, dtype float):
#                 'initial Mask' vector.
#                 This vector initializes the only optimized parameter.
#                 The initial cluster inclusion vector. Masks which clusters
#                  are included in the prediction.
#                 shape: (n_clusters)
#                 If None, then vector is initialized as small random values.
#             device (str):
#                 The device to use for the computation. ('cpu', 'cuda', etc.)
#             optimizer_partial (torch.optim.Optimizer):
#                 A torch optimizer with all but the parameters initialized.
#                 If None, then the optimizer is initialized with Adam and 
#                  some default parameters.
#                 This can be made using:
#                  functools.partial(torch.optim.Adam, lr=x, betas=(x,x)).
#             scheduler (torch.optim.lr_scheduler):
#                 A torch learning rate scheduler.
#             dmCEL_temp (float):
#                 ===IMPORTANT HYPERPARAMETER===
#                 The temperature used in the cross entropy loss of the
#                  interaction matrix.
#             dmCEL_sigSlope (float):
#                 The slope of the sigmoid used to constrain and activate the
#                  m vector. Higher values result in more forced certainty, 
#                  but less stability.
#                 Recommend staying as low as possible, around 1-5.
#             dmCEL_sigCenter (float):
#                 The center of the sigmoid described above.
#                 Recommend keeping to 0.5.
#             dmCEL_penalty (float):
#                 The penalty applied to the cross entropy loss of the 
#                  interaction matrix. Best to keep this to 1 and change the
#                  other penalties.
#             sampleWeight_softplusKwargs (dict):
#                 The kwargs passed to the softplus function used to penalize
#                  when samples are included more than once. 
#                 Recommend keeping this around default values.
#             sampleWeight_penalty (float):
#                 ===IMPORTANT HYPERPARAMETER===
#                 The penalty applied when samples are included more than once.
#             fracWeighted_goalFrac (float):
#                 ===IMPORTANT HYPERPARAMETER===
#                 The goal fraction of samples to be included in the output.
#                 Recommend keeping this to 1 for most applications.
#             fracWeighted_sigSlope (float):
#                 The slope of the sigmoid used to constrain the sample weights
#                  so as to give a score simply on whether or not they were
#                  included.
#                 Recommend keeping this to 1-5.
#             fracWeighted_sigCenter (float):
#                 The center of the sigmoid described above.
#                 Recommend keeping to 0.5.
#             fracWeight_penalty (float):
#                 The penalty applied to the sample weights loss.
#             maskL1_penalty (float):
#                 ===IMPORTANT HYPERPARAMETER===
#                 The penalty applied to the L1 norm of the mask.
#                 Adjust this to reduce the probability of extracting clusters
#                  with low scores.
#             tol_convergence (float):
#                 The tolerance for the convergence of the optimization.
#             window_convergence (int):
#                 The number of iterations to use in the convergence check.
#                 A regression is performed on the last window_convergence to
#                  see if the best fit line has changed by less than
#                  tol_convergence.
#             freqCheck_convergence (int):
#                 The number of iterations to wait betwee checking for
#                  convergence.
#             verbose (bool):
#                 Whether to print progress.

#         Attributes set:
#             self.c (torch_sparse.SparseTensor):
#                 The cluster similarity matrix.
#             self.h (torch.SparseTensor):
#                 The cluster membership matrix.
#             self.w (torch.SparseTensor):
#                 The cluster weights vector.
#             self.m (torch.Tensor):
#                 The cluster inclusion vector.
#         """
#         import torch_sparse as ts
#         import scipy.sparse

#         ## Imports
#         super().__init__()

#         ## c.shape[0] == c.shape[1] == h.shape[1]
#         assert c.shape[0] == c.shape[1] == h.shape[1], 'RH ERROR: the following must be true:  c.shape[0] == c.shape[1] == h.shape[1], they should all be n_clusters'

#         self._n_clusters = h.shape[1]

#         self._DEVICE = device

#         ## c must be cast as a torch_sparse.SparseTensor because the masking 
#         ##  operation (c * m[None,:]) is not implemented in standard torch.sparse yet...
#         c_tmp = indexing.scipy_sparse_to_torch_coo(c).coalesce().type(torch.float32)
#         self.c = ts.SparseTensor(
#             row=c_tmp.indices()[0], 
#             col=c_tmp.indices()[1], 
#             value=c_tmp.values(),
#             sparse_sizes=c_tmp.shape,
#         ).to(self._DEVICE)

#         self.h = indexing.scipy_sparse_to_torch_coo(h).coalesce().type(torch.float32).to(self._DEVICE)

#         ## w is converted to a square diagonal matrix so that h @ w masks the columns of h (torch.sparse doesn't support h * w[None,:] yet...)
#         w_tmp = scipy.sparse.lil_matrix((self._n_clusters, self._n_clusters))
#         w_tmp[range(self._n_clusters), range(self._n_clusters)] = (w / w.max()) if w is not None else 1
#         self.w = indexing.scipy_sparse_to_torch_coo(w_tmp).coalesce().type(torch.float32).to(self._DEVICE)

#         ## optimization seems to proceed best when values initialize as small random values. New ideas welcome here.
#         self.m = m_init.to(self._DEVICE) if m_init is not None else (torch.ones(self._n_clusters)*0.1 + torch.rand(self._n_clusters)*0.05).type(torch.float32).to(self._DEVICE)
#         self.m.requires_grad=True

#         self._dmCEL_penalty = dmCEL_penalty
#         self._sampleWeight_penalty = sampleWeight_penalty
#         self._fracWeight_penalty = fracWeight_penalty
#         self._maskL1_penalty = maskL1_penalty

#         self._dmCEL_temp = torch.as_tensor([dmCEL_temp], dtype=torch.float32).to(self._DEVICE)
#         self._dmCEL_sigSlope = torch.as_tensor([dmCEL_sigSlope], dtype=torch.float32).to(self._DEVICE)
#         self._dmCEL_sigCenter = torch.as_tensor([dmCEL_sigCenter], dtype=torch.float32).to(self._DEVICE)
#         self._fracWeighted_sigSlope = torch.as_tensor([fracWeighted_sigSlope], dtype=torch.float32).to(self._DEVICE)
#         self._fracWeighted_sigCenter = torch.as_tensor([fracWeighted_sigCenter], dtype=torch.float32).to(self._DEVICE)


#         # self._dmCEL_penalty  = torch.as_tensor([self._dmCEL_penalty]).to(self._DEVICE)
#         # self._sampleWeight_penalty = torch.as_tensor([self._sampleWeight_penalty]).to(self._DEVICE)
#         # self._fracWeight_penalty = torch.as_tensor([self._fracWeight_penalty]).to(self._DEVICE)
#         # self._maskL1_penalty = torch.as_tensor([self._maskL1_penalty]).to(self._DEVICE)

#         # self._dmCEL_penalty.requires_grad = True
#         # self._sampleWeight_penalty.requires_grad = True
#         # self._fracWeight_penalty.requires_grad = True
#         # self._maskL1_penalty.requires_grad = True

#         # self._dmCEL_temp.requires_grad = True
#         # self._dmCEL_sigSlope.requires_grad = True
#         # self._dmCEL_sigCenter.requires_grad = True
#         # self._fracWeighted_sigSlope.requires_grad = True
#         # self._fracWeighted_sigCenter.requires_grad = True

#         self._optimizer = optimizer_partial(params=[self.m]) if optimizer_partial is not None else torch.optim.Adam(params=[self.m], lr=1e-2, betas=(0.9, 0.900))
#         # self._optimizer = optimizer_partial(params=[self.m, self._dmCEL_temp, self._dmCEL_sigSlope, self._dmCEL_sigCenter, self._fracWeighted_sigSlope, self._fracWeighted_sigCenter]) if optimizer_partial is not None else torch.optim.Adam(params=[self.m], lr=1e-2, betas=(0.9, 0.900))
#         self._scheduler = scheduler_partial(optimizer=self._optimizer) if scheduler_partial is not None else torch.optim.lr_scheduler.LambdaLR(optimizer=self._optimizer, lr_lambda=lambda x : x, last_epoch=-1, verbose=True)
        
#         self._dmCEL = self._DoubleMasked_CEL(
#             c=self.c,
#             device=self._DEVICE,
#             temp=self._dmCEL_temp,
#             sig_slope=self._dmCEL_sigSlope,
#             sig_center=self._dmCEL_sigCenter,
#         )

#         self._loss_fracWeighted = self._Loss_fracWeighted(
#             h=self.h,
#             w=self.w,
#             goal_frac=fracWeighted_goalFrac,
#             sig_slope=self._fracWeighted_sigSlope,
#             sig_center=self._fracWeighted_sigCenter,
#         )

#         self._loss_sampleWeight = self._Loss_sampleWeight(
#             softplus_kwargs=sampleWeight_softplusKwargs,
#         )

#         self._convergence_checker = self._Convergence_checker(
#             tol_convergence=tol_convergence,
#             window_convergence=window_convergence,
#         )

#         self._tol_convergence = tol_convergence
#         self._window_convergence = window_convergence
#         self._freqCheck_convergence = freqCheck_convergence
#         self._verbose = verbose

#         self._i_iter = 0
#         self.losses_logger = {
#             'loss': [],
#             'L_cs': [],
#             'L_fracWeighted': [],
#             'L_sampleWeight': [],
#             'L_maskL1': [],
#         }

#         ## clean up. Some of the initialization steps generate intermediate tensors
#         ##  that are not needed long term.
#         gc.collect()
#         torch.cuda.empty_cache()

#     def fit(
#         self, 
#         min_iter=1e3,
#         max_iter=1e4,
#         verbose=True, 
#         verbose_interval=100
#     ):
#         """
#         Fit the model.
#         This method can be interupted during a run and restarted without issues.

#         Uses the standard PyTorch gradient descent approach:
#             1. zero the parameter gradients (optimizer.zero_grad())
#             2. compute the loss (loss=...)
#             3. compute the gradients of the loss w.r.t. the parameters (loss.backward())
#             4. update the parameters (optimizer.step())

#         There are 4 components to the loss:
#             1. L_cs: Penalizes how similar the unmasked clusters are relative to each other.
#             2. L_fracWeighted: Penalizes how many samples are not assigned to a cluster.
#             3. L_sampleWeight: Penalizes samples that are assigned to multiple clusters.
#             4. L_maskL1: Penalizes using many (instead of few) clusters to mask the samples.

#         Issue handling things:
#             - clamping the values of m to > -14. This is to prevent a dead gradient on those
#              parameter values.
#             - checking if loss is NaN. If it is, the optimization is terminated before backward pass.

#         The convergence checker fits a line to the last self._window_convergence iterations of self.losses_logger
#          and checks if the slope of the line is less than self._tol_convergence.

#         Args:
#             min_iter (int): 
#                 The minimum number of iterations to run.
#             max_iter (int): 
#                 The maximum number of iterations to run.
#             verbose (bool): 
#                 If True, print the loss every verbose_interval iterations.
#             verbose_interval (int): 
#                 The interval at which to print the loss.
#         """
#         loss_smooth = np.nan
#         diff_window_convergence = np.nan
#         while self._i_iter <= max_iter:
#             self._optimizer.zero_grad()

#             L_cs = self._dmCEL(c=self.c, m=self.m, w=self.w) * self._dmCEL_penalty  ## 'cluster similarity loss'
#             L_sampleWeight = self._loss_sampleWeight(self.h, self.activate_m()) * self._sampleWeight_penalty ## 'sample weight loss'
#             L_fracWeighted = self._loss_fracWeighted(self.activate_m()) * self._fracWeight_penalty ## 'fraction weighted loss'
#             L_maskL1 = torch.sum(torch.abs(self.activate_m())) * self._maskL1_penalty ## 'L1 on mask loss'

#             self._loss = L_cs + L_fracWeighted + L_sampleWeight + L_maskL1 ## 'total loss'

#             if torch.isnan(self._loss):
#                 print(f'STOPPING EARLY: loss is NaN. iter: {self._i_iter}  loss: {self._loss.item():.4f}  L_cs: {L_cs.item():.4f}  L_fracWeighted: {L_fracWeighted.item():.4f}  L_sampleWeight: {L_sampleWeight.item():.4f}  L_maskL1: {L_maskL1.item():.4f}')
#                 break

#             self._loss.backward()
#             self._optimizer.step()
#             self._scheduler.step()

#             self.m.data = torch.maximum(self.m.data , torch.as_tensor(-14, device=self._DEVICE)) ## clamp m values to prevent zeroing out and a dead gradient

#             ## populate logger with values
#             self.losses_logger['loss'].append(self._loss.item())
#             self.losses_logger['L_cs'].append(L_cs.item())
#             self.losses_logger['L_fracWeighted'].append(L_fracWeighted.item())
#             self.losses_logger['L_sampleWeight'].append(L_sampleWeight.item())
#             self.losses_logger['L_maskL1'].append(L_maskL1.item())

#             ## check for convergence
#             if self._i_iter%self._freqCheck_convergence==0 and self._i_iter>self._window_convergence and self._i_iter>min_iter:
#                 diff_window_convergence, loss_smooth, converged = self._convergence_checker(self.losses_logger['loss'])
#                 if converged:
#                     print(f"STOPPING: Convergence reached in {self._i_iter} iterations.  loss: {self.losses_logger['loss'][-1]:.4f}  loss_smooth: {loss_smooth:.4f}")
#                     break
            
#             ## print loss values
#             if verbose and self._i_iter % verbose_interval == 0:
#                 print(f'iter: {self._i_iter}:  loss_total: {self._loss.item():.4f}  lr: {self._scheduler.get_last_lr()[0]:.5f}   loss_cs: {L_cs.item():.4f}  loss_fracWeighted: {L_fracWeighted.item():.4f}  loss_sampleWeight: {L_sampleWeight.item():.4f}  loss_maskL1: {L_maskL1.item():.4f}  diff_loss: {diff_window_convergence:.4f}  loss_smooth: {loss_smooth:.4f}')
#             self._i_iter += 1


#     def predict(
#         self,
#         m_threshold=0.5
#     ):   
#         """
#         Return predicted cluster assignments based on a threshold on the
#          activated 'm' vector.
#         It can be useful to first plot the activated 'm' vector to see the
#          distribution: self.plot_clusterWeights().
        
#         Args:
#             m_threshold (float): 
#                 Threshold on the activated 'm' vector.
#                 Clusters with activated 'm' values above the threshold are
#                  assigned a cluster ID.
#         """
#         h_ts = indexing.torch_to_torchSparse(self.h)

#         self.m_bool = (self.activate_m() > m_threshold).squeeze().cpu()
#         h_ts_bool = h_ts[:, self.m_bool]
#         n_multiples = (h_ts_bool.sum(1) > 1).sum()
#         print(f'WARNING: {n_multiples} samples are matched with multiple clusters. Consider increasing the sample_weight_penalty during training.') if n_multiples > 0 else None
#         h_preds = (h_ts_bool * (torch.arange(self.m_bool.sum(), device=self._DEVICE)[None,:]+1)).detach().cpu()

#         if h_preds.numel() == 0:
#             print(f'WARNING: No predictions made.  m_threshold: {m_threshold}')
#             return None, None

#         preds = h_preds.max(dim=1) - 1
#         preds[torch.isinf(preds)] = -1
        
#         h_m = (h_ts * self.activate_m()[None,:]).detach().cpu().to_dense()
#         confidence = h_m.var(1) / h_m.mean(1)

#         self.scores_clusters = torch_helpers.diag_sparse(self.w).cpu()[self.m_bool]
#         self.scores_samples = (h_preds * self.scores_clusters[None,:]).max(1)
        
#         self.preds = preds
#         self.confidence = confidence
        
#         return self.preds, self.confidence, self.scores_samples, self.m_bool

#     def activate_m(self):
#         """
#         Pass the 'm' ('masking', the only optimized parameter) through the
#          activation function and return it.
#         """
#         return self._dmCEL.activation(self.m)


#     class _DoubleMasked_CEL:
#         """
#         'Double Masked Cross-Entropy Loss' Class. Derived here.
#         Gives a loss value for how well-separated a set of clusters are.
#         Input is a cluster similarity matrix (c, shape: (n_clusters, n_clusters)),
#          and a mask vector (m, shape: (n_clusters,)).
#         Output is a loss value.
#         Operation proceeds as follows:
#         1. Pass mask vector through activation function. ma=sigmoid(m)
#         2. Compute the loss value for each cluster.
#             a. Mask the columns of 'c' with 'ma': cm=c*ma[None,:]
#             b. Replace the diagonal values of 'cm' with the diagonal values of 'c'
#             c. Compute the cross-entropy loss value for each cluster. lv=CEL(cm, labels=arange(n_clusters))
#         3. Weight the loss values by the mask vector and do a mean reduction. l=lv@ma
        
#         RH 2022
#         """
#         def __init__(
#             self,
#             c,
#             device='cpu',
#             temp=1,
#             sig_slope=5,
#             sig_center=0.5,
#         ):    
#             """
#             Initializes the 'Double Masked Cross-Entropy Loss' Class.

#             Args:
#                 c (torch_sparse.SparseTensor):
#                     Cluster similarity matrix (c, shape: (n_clusters, n_clusters)).
#                 device (str):
#                     Device to use for the computation.
#                 temp (float):
#                     Temperature parameter for the softmax activation function.
#                     Lower values results in more non-convexity but stronger convergence (steeper gradient).
#                 sig_slope (float):
#                     Slope of the sigmoid activation function of the mask vector.
#                 sig_center (float):
#                     Center of the sigmoid activation function of the mask vector.
#             """
#             self._n_clusters = c.sizes()[0]

#             self.temp = temp
            
#             ## Cross-Entropy Loss: since we are trying to maximize the diagonal, CEL can be simplified as:
#             ##  -diag(log_softmax(x))
#             self.CEL = lambda x :  - ts_logSoftmax(x, temperature=self.temp, shift=None).get_diag() ## this function ts_logSoftmax existed at one point
            
#             ## We will be activating the optimized 'm' parameter to scale it between 0 and 1.
#             self.activation = self.make_sigmoid_function(sig_slope, sig_center)

#             self.device=device
            
#             ## We define a best and worst case CEL value so that we can scale the loss to be invariant to
#             ##  things like temperature and number of clusters.
#             self.worst_case_loss = self.CEL(c)
#             self.best_case_loss = self.CEL(
#                 ts.tensor.SparseTensor(
#                     row=torch.arange(self._n_clusters, device=device, dtype=torch.int64),
#                     col=torch.arange(self._n_clusters, device=device, dtype=torch.int64),
#                     value=c.get_diag().type(torch.float32)**0,
#                     sparse_sizes=(self._n_clusters, self._n_clusters),
#                 )
#             ) ## the best case is just a diagonal matrix with the same values as the input matrix.
#             self.worst_minus_best = self.worst_case_loss - self.best_case_loss  + 1e-8  ## add a small amount for numerical stability.

#         def make_sigmoid_function(
#             self,
#             sig_slope=5,
#             sig_center=0.5,
#         ):
#             return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))
            
#         def __call__(self, c, m, w):
#             import torch_sparse as ts
#             import scipy.sparse

#             ma = self.activation(m)  ## 'activated mask'. Constrained to be 0-1

#             ## Below gives us our loss for each column, which will later be scaled by 'ma' 
#             ##  to get the final loss.
#             ## The idea is that 'ma' masks out columns in 'c' (while ignoring elements along
#             ##  the diagonal) in order to maximize the diagonal elements relative to all other
#             ##  elements in the row. Optimization should result in 'selection' of clusters by
#             ##  masking out poor clusters that are too similar to good clusters.
#             lv = self.CEL(
#                 ts_setDiag_lowMem(
#                     c * ma[None,:],
#                     c.get_diag()**0
#                 )  ## this function existed at one point too
#             ) ## 'loss vector'

#             lv_norm = (lv - self.best_case_loss) / self.worst_minus_best  ## scale the loss of each cluster by the best and worst case possibilities.
            
#             ## weight the loss of each clusters by the activated masking parameter 'ma'
#             l = (lv_norm @ ma) / ma.sum()  ## 'loss'

#             return l


#     class _Loss_fracWeighted:
#         """
#         'Fraction Weighted Loss' Class. Derived here.
#         Gives a loss value for the fraction of clusters that are highly weighted
#          scaled by how valuable each cluster is.
#         Inputs are the cluster membership ('multiHot') matrix (h, shape: (n_clusters, n_samples)),
#          and the cluster scores ('weighting') vector (w, shape: (n_clusters,)).
#         Output is a loss value.
#         Operation proceeds as follows:
#         0. In initialization, weight the h matrix by the w vector. hw=h*w[None,:]
#         1. Mask and sum-reduce the hw matrix with the mask vector (dot product) to get the
#          score-weighted penalty for excluding that cluster. hwm=hw@m (hwm shape: n_samples)
#         2. Pass hwm through an activation function. hwma=sigmoid(hwm)
#         3. Take mean squared error with respect to the goal number of samples positively
#          weighted. l=(hwma-g)^2

#         RH 2022
#         """
#         def __init__(
#             self,
#             h,
#             w,
#             goal_frac=1.0,
#             sig_slope=5,
#             sig_center=0.5,
#         ):
#             # self.h_w = h.type(torch.float32) * w[None,:]
#             self.h_w = h.type(torch.float32) @ w
#             # self.h_w = h.type(torch.float32)
        
#             self.goal_frac = goal_frac
#             self.sigmoid = self.make_sigmoid_function(sig_slope, sig_center)
            
#         def make_sigmoid_function(
#             self,
#             sig_slope=5,
#             sig_center=0.5,
#         ):
#             return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))

#         def generate_sampleWeights(self, m):
#             return self.h_w @ m
#         def activate_sampleWeights(self, sampleWeights):
#             return self.sigmoid(sampleWeights)

#         def __call__(self, m):
#             return ((self.activate_sampleWeights(self.generate_sampleWeights(m))).mean() - self.goal_frac)**2

#     class _Loss_sampleWeight:
#         def __init__(
#             self,
#             softplus_kwargs=None,
#         ):
#             self.softplus = torch.nn.Softplus(**softplus_kwargs)
#             self.weight_high = 1
#             self.offset = self.activate_sampleWeights(torch.ones(1, dtype=torch.float32)).max()
        
#         def generate_sampleWeights(self, h, m):
#             return h @ m
#         def activate_sampleWeights(self, sampleWeights):
#             return self.softplus(sampleWeights - 1 - 4/self.softplus.beta)*self.weight_high

#         def __call__(self, h, m):
#             return (self.activate_sampleWeights(self.generate_sampleWeights(h, m))).mean()

#     class _Convergence_checker:
#         """
#         'Convergence Checker' Class.
#         Checks for convergence of the optimization. Uses Ordinary Least Squares (OLS) to 
#          fit a line to the last 'window_convergence' number of iterations.
#         """
#         def __init__(
#             self,
#             tol_convergence=1e-2,
#             window_convergence=100,
#         ):
#             """
#             Initialize the convergence checker.
            
#             Args:
#                 tol_convergence (float): 
#                     Tolerance for convergence.
#                     Corresponds to the slope of the line that is fit.
#                 window_convergence (int):
#                     Number of iterations to use for fitting the line.
#             """
#             self.window_convergence = window_convergence
#             self.tol_convergence = tol_convergence

#             self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

#         def OLS(self, y):
#             """
#             Ordinary least squares.
#             Fits a line to y.
#             """
#             X = self.line_regressor
#             theta = torch.inverse(X.T @ X) @ X.T @ y
#             y_rec = X @ theta
#             bias = theta[-1]
#             theta = theta[:-1]

#             return theta, y_rec, bias

#         def __call__(
#             self,
#             loss_history,
#         ):
#             """
#             Forward pass of the convergence checker.
#             Checks if the last 'window_convergence' number of iterations are
#              within 'tol_convergence' of the line fit.

#             Args:
#                 loss_history (list):
#                     List of loss values for the last 'window_convergence' number of iterations.

#             Returns:
#                 diff_window_convergence (float):
#                     Difference of the fit line over the range of 'window_convergence'.
#                 loss_smooth (float):
#                     The mean loss over 'window_convergence'.
#                 converged (bool):
#                     True if the 'diff_window_convergence' is less than 'tol_convergence'.
#             """
#             if len(loss_history) < self.window_convergence:
#                 return torch.nan, torch.nan, False
#             loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
#             theta, y_rec, bias = self.OLS(y=loss_window)

#             diff_window_convergence = (y_rec[-1] - y_rec[0])
#             loss_smooth = loss_window.mean()
#             converged = True if torch.abs(diff_window_convergence) < self.tol_convergence else False
#             return diff_window_convergence.item(), loss_smooth.item(), converged
            

#     def plot_loss(self):
#         plt.figure()
#         plt.plot(self.losses_logger['loss'], linewidth=4)
#         plt.plot(self.losses_logger['L_cs'])
#         plt.plot(self.losses_logger['L_fracWeighted'])
#         plt.plot(self.losses_logger['L_sampleWeight'])
#         plt.plot(self.losses_logger['L_maskL1'])

#         plt.legend(self.losses_logger.keys())
#         plt.xlabel('iteration')
#         plt.ylabel('loss')

#     def plot_clusterWeights(self, plot_raw_m=False):
#         plt.figure()
#         if plot_raw_m:
#             plt.hist(self.m.detach().cpu().numpy(), bins=50)
#         plt.hist(self.activate_m().detach().cpu().numpy(), bins=50)
#         plt.xlabel('cluster weight')
#         plt.ylabel('count')

#     def plot_clusterScores(self, bins=100):
#         if hasattr(self, 'm_bool'):
#             m_bool = self.m_bool
#             confidence = self.confidence
#             preds = self.preds
#         else:
#             preds, confidence, scores_samples, m_bool = self.predict()
#             if preds is None:
#                 print('Plot failed: preds is None.')
#                 return None
#         scores = helpers.diag_sparse(self.w).cpu()[m_bool.cpu()]

#         plt.figure()
#         plt.hist(scores, bins=bins, log=True)
#         plt.xlabel('cluster score')
#         plt.ylabel('count')

#         plt.figure()
#         plt.hist(confidence[preds >= 0], bins=bins, log=True)
#         plt.xlabel('confidence')
#         plt.ylabel('count')

#     def plot_sampleWeights(self):
#         sampleWeights = self._loss_sampleWeight.generate_sampleWeights(self.h, self.activate_m()).detach().cpu().numpy()

#         plt.figure()
#         plt.hist(sampleWeights, bins=50)
#         plt.xlabel('sample weight')
#         plt.ylabel('count')

#     def plot_labelCounts(self):
#         if hasattr(self, 'preds'):
#             preds = self.preds
#         else:
#             preds, confidence, scores_samples, m_bool = self.predict()
#             if preds is None:
#                 print('Skipping plot_labelCounts: preds is None.')
#                 return None
            
#         labels_unique, label_counts = np.unique(preds, return_counts=True)

#         fig, axs = plt.subplots(1, 2, figsize=(10,5))
#         axs[0].bar(labels_unique, label_counts)
#         axs[0].set_xlabel('label')
#         axs[0].set_ylabel('count')
#         axs[1].hist(label_counts[labels_unique>=0], bins=25)
#         axs[1].set_xlabel('n_roi per cluster')
#         axs[1].set_ylabel('counts')

#         return fig, axs

#     def plot_c_threshold_matrix(self, m_threshold=0.5):
#         mt = self.activate_m() > m_threshold
#         plt.figure()
#         plt.imshow(self.c[mt, mt].detach().cpu(), aspect='auto')

#     def plot_c_masked_matrix(self, m_threshold=0.5, **kwargs_imshow):
#         import sparse
#         mt = self.activate_m() > m_threshold
#         # return (self.c * mt[None,:])
#         cdm = (sparse.COO(self.c.to_scipy()) * mt[None,:].cpu().numpy()  * mt[:, None].cpu().numpy()).tocsr().tolil()
#         cdm[list(self._dmCEL.idx_diag.cpu().numpy()), list(self._dmCEL.idx_diag.cpu().numpy())] = self.c.get_diag().cpu()
#         plt.figure()
#         plt.imshow(cdm.toarray(), aspect='auto')


# class rich_clust():
#     """
#     Basic clustering algorithm based on gradient descent.
#     The approach is to optimize a 'multihot matrix' (h),
#      which is an n_samples x n_clusters matrix describing
#      the membership of each sample to each cluster. This matrix
#      is used to mask a similarity matrix (s); allowing for the
#      calculation of the mean pairwise similarity within each 
#      cluster and between each pair of clusters. The loss
#      function maximizes the within similarity and minimizes the
#      between similarity.
#     Functionally, the result ends up similar to k-means
#      clustering.
#     This method is not great on its own, but is useful because
#      it allows for custom penalties to be applied to the loss.
#     It struggles due to a non-convex solution space which
#      results in sensitivity to initial conditions and weird 
#      sensitivities to hyperparameters.
#     RH 2022
#     """

#     def __init__(
#         self,
#         s,
#         n_clusters=2,
#         l=1,
#         temp_h=1,
#         temp_c=1,
#         optimizer=None,
#         DEVICE='cpu',
#         init_h=None,
#     ):
#         """
#         Args:
#             s (torch.Tensor):
#                 Similarity matrix.
#                 shape: (n_samples, n_samples)
#                 dtype: torch.float32
#                 The diagonal should be zeros.
#                 Example:
#                     d = torch.cdist(data, data, p=2).type(torch.float32)
#                     s = torch.maximum(1-d, torch.as_tensor([0]))
#                     s = s * torch.logical_not(torch.eye(s.shape[0]))
#             n_clusters (int):
#                 The number of clusters to find.
#             l (float):
#                 'Locality' parameter. The exponent applied to 
#                   the similarity matrix. Higher values allow for
#                   more non-convex clusters, but can result in 
#                   instability and cluster-splitting.
#             temp_h (float):
#                 The temperature for the multihot matrix. Higher
#                  values result in fuzzier cluster edges.
#             temp_c (float):
#                 The temperature for the cluster membership matrix.
#                 Higher values result in globally less confident
#                  cluster scores.
#             optimizer (partial torch.optim.Optimizer):
#                 Optional. If None, then Adam is used.
#                 The partial optimizer to use.
#                 Can be constructed like: 
#                     functools.partial(torch.optim.Adam, lr=0.01, weight_decay=0.00001)
#             DEVICE (str):
#                 The device to use. Default is 'cpu'.
#             init_h (torch.Tensor):
#                 The initial multihot matrix. If None, then
#                  random initialization is used.
#         """
    
#         self.s = s.to(DEVICE)**l
#         self.n_clusters = n_clusters

#         self.l = l
#         self.temp_h = temp_h
#         self.temp_c = temp_c

#         self.DEVICE = DEVICE

#         if init_h is None:
#             self.h = self._initialize_multihot_matrix(self.s.shape[0], self.n_clusters).to(self.DEVICE)
#         else:
#             self.h = init_h.to(self.DEVICE)
#         self.h.requires_grad = True

#         self.ii_normFactor = lambda i   : i * (i-1)
#         self.ij_normFactor = lambda i,j : i * j

#         if optimizer is None:
#             self.optimizer = torch.optim.Adam(
#                 [self.h], 
#                 lr=0.1,
#                 # weight_decay=1*10**-6
#             )
#         else:
#             self.optimizer = optimizer(params=[self.h])


#     def _initialize_multihot_matrix(self, n_samples, n_clusters):
#         h = torch.rand(size=(n_samples, n_clusters))  ## Random initialization
#         return h
    
#     def _make_cluster_similarity_matrix(self, s, h, temp_h, DEVICE='cpu'):
#         h = torch.nn.functional.softmax(h/temp_h, dim=1)
#     #     return torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h))  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#     #     return (  torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h)) * torch.eye(h.shape[1]).to(DEVICE)  +  torch.einsum('ab, abcd -> cd', s**4, torch.einsum('ac, bd -> abcd', h,h)) * (torch.logical_not(torch.eye(h.shape[1]).to(DEVICE)))*0.05  )  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#         return torch.einsum('ab, ac, bd -> cd', s, h, h)  /  \
#             ( (torch.eye(h.shape[1]).to(DEVICE) * self.ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * self.ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )

#     def fit(self, n_iter=200):

#         for i_iter in range(n_iter):
#             self.optimizer.zero_grad()

#             self.c = self._make_cluster_similarity_matrix(
#                 s=self.s,
#                 h=self.h, 
#                 temp_h=self.temp_h,
#                 DEVICE=self.DEVICE
#             )
            
#             self.L_cs = torch.nn.functional.cross_entropy(self.c/self.temp_c, torch.arange(self.n_clusters).to(self.DEVICE))
#             self.loss = self.L_cs
#             self.loss.backward()
#             self.optimizer.step()
