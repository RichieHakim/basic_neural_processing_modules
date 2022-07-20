import sklearn
import numpy as np

import torch

import copy

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


# class rich_clust2():
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
#         n_wildClusters=0,
#         penalty_wildClusters=0,
#         l=1,
#         p_nSpC=None,
#         temp_h=1,
#         temp_c=1,
#         optimizer=None,
#         grad_clip=None,
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
#             n_wildClusters (int):
#                 The number of clusters that are not subject to the 
#                  nSpC penalty.
#             penalty_wildClusters (float):
#                 The penalty for any weight existing in a wild cluster
#             l (float):
#                 'Locality' parameter. The exponent applied to 
#                   the similarity matrix. Higher values allow for
#                   more non-convex clusters, but can result in 
#                   instability and cluster-splitting.
#             p_nSpC (function):
#                 'penalty: number of samples per cluster'
#                 This function is used to apply a penalty to the
#                  number of samples per cluster.
#                 Example:
#                     p_nSpC = lambda x : (torch.log(-(x-(nSpC+1)))**(4)).sum()
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
#             grad_clip (float):
#                 Optional. If None, then no gradient clipping is used.
#                 The maximum gradient norm.
#             DEVICE (str):
#                 The device to use. Default is 'cpu'.
#             init_h (torch.Tensor):
#                 The initial multihot matrix. If None, then
#                  random initialization is used.
#         """
    
#         self.s = s.to(DEVICE)**l
#         self.n_clusters = n_clusters
#         self.n_wildClusters = n_wildClusters

#         self.penalty_wildClusters = penalty_wildClusters
#         self.l = l
#         self.temp_h = temp_h
#         self.temp_c = temp_c

#         self.DEVICE = DEVICE

#         if p_nSpC is None:
#             # nSpC = s.shape[0] / self.n_clusters
#             # self.p_nSpC = lambda x : torch.log(-(x-(nSpC+1)))**(4)
#             self.p_nSpC = lambda x : x*0
#         else:
#             self.p_nSpC = p_nSpC

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

#         self.grad_clip = grad_clip

#         self.losses = {
#             'L_cs': [],
#             'L_nSpC': [],
#             'L_wildClusters': [],
#             'L_total': [],
#         }

#         self.i_iter = 0


#     def _initialize_multihot_matrix(self, n_samples, n_clusters):
#         # h = torch.rand(size=(n_samples, n_clusters))  ## Random initialization
#         # h = h / h.sum(dim=0, keepdim=True)

#         h = torch.cat(
#             (
#                 torch.rand(n_samples,n_clusters) * 0.1, 
#                 # (torch.ones(n_samples,1)) * 1
#             ),
#             dim=1,
#         )
#         return h
    
#     def _make_cluster_similarity_matrix(self, s, h, DEVICE='cpu'):
#         # h = torch.nn.functional.softmax(h/temp_h, dim=1)
#     #     return torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h))  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#     #     return (  torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h)) * torch.eye(h.shape[1]).to(DEVICE)  +  torch.einsum('ab, abcd -> cd', s**4, torch.einsum('ac, bd -> abcd', h,h)) * (torch.logical_not(torch.eye(h.shape[1]).to(DEVICE)))*0.05  )  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#         return torch.einsum('ab, ac, bd -> cd', s, h, h)  /  \
#             ( (torch.eye(h.shape[1]).to(DEVICE) * self.ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * self.ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )

#     def fit(
#         self,
#         n_iter=200,
#         verbose=False,
#         verbose_interval=10,
#     ):

#         for _ in range(n_iter):
#             self.optimizer.zero_grad()

#             self.h_p = torch.nn.functional.softmax(self.h/self.temp_h, dim=1)
#             # self.h_p = torch.nn.functional.softplus(self.h/self.temp_h, beta=50, threshold=10)
#             self.c = self._make_cluster_similarity_matrix(
#                 s=self.s,
#                 h=self.h_p, 
#                 DEVICE=self.DEVICE
#             )
#             # print(self.h_p.sum(0))
#             # print(self.p_nSpC(self.h_p.sum(0)))
            
#             self.L_cs = torch.nn.functional.cross_entropy(self.c/self.temp_c, torch.arange(self.n_clusters).to(self.DEVICE))
#             self.L_nSpC = self.p_nSpC(self.h_p.sum(0)[:-self.n_wildClusters]).mean()
#             # print(self.h_p.sum(0)[:-self.n_wildClusters])
#             # print(self.p_nSpC(self.h_p.sum(0)[:-self.n_wildClusters]))
#             self.L_wildClusters = (self.h_p.mean(0)[-self.n_wildClusters:]*self.penalty_wildClusters).mean()
#             self.loss = self.L_cs + self.L_nSpC + self.L_wildClusters

#             self.loss.backward()
#             if self.grad_clip is not None:
#                 torch.nn.utils.clip_grad_norm_(self.h, self.grad_clip)
#             self.optimizer.step()

#             self.losses['L_cs'].append(self.L_cs.item())
#             self.losses['L_nSpC'].append(self.L_nSpC.item())
#             self.losses['L_wildClusters'].append(self.L_wildClusters.item())
#             self.losses['L_total'].append(self.loss.item())

#             if verbose and self.i_iter % verbose_interval == 0:
#                 print(f'iter: {self.i_iter}:  loss_total: {self.loss.item():.4f}   loss_cs: {self.L_cs.item():.4f}  loss_nSpC: {self.L_nSpC.item():.4f}  loss_wildClusters: {self.L_wildClusters.item():.4f}')

#             # if self.loss.item() > 10:
#             #     break
#             self.i_iter += 1
            

# class rich_clust3():
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
#         p=None,
#         n_clusters=2,
#         n_wildClusters=0,
#         penalty_wildClusters=0,
#         penalty_ante=1,
#         l=1,
#         p_nSpC=None,
#         temp_h=1,
#         temp_c=1,
#         optimizer=None,
#         grad_clip=None,
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
#             p (torch.Tensor):
#                 Penalty matrix.
#                 shape: (n_samples, n_samples)
#                 dtype: torch.float32
#                 This is the penalty for two ROIs being assigned to the
#                  same cluster.
#             n_clusters (int):
#                 The number of clusters to find.
#             n_wildClusters (int):
#                 The number of clusters that are not subject to the 
#                  nSpC penalty.
#             penalty_wildClusters (float):
#                 The penalty for any weight existing in a wild cluster
#             penalty_ante (float):
#                 The penalty for weight existing in a normal cluster
#             l (float):
#                 'Locality' parameter. The exponent applied to 
#                   the similarity matrix. Higher values allow for
#                   more non-convex clusters, but can result in 
#                   instability and cluster-splitting.
#             p_nSpC (function):
#                 'penalty: number of samples per cluster'
#                 This function is used to apply a penalty to the
#                  number of samples per cluster.
#                 Example:
#                     p_nSpC = lambda x : (torch.log(-(x-(nSpC+1)))**(4)).sum()
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
#             grad_clip (float):
#                 Optional. If None, then no gradient clipping is used.
#                 The maximum gradient norm.
#             DEVICE (str):
#                 The device to use. Default is 'cpu'.
#             init_h (torch.Tensor):
#                 The initial multihot matrix. If None, then
#                  random initialization is used.
#         """
    
#         self.s = s.to(DEVICE)**l
#         self.n_clusters = n_clusters
#         self.n_wildClusters = n_wildClusters

#         self.penalty_wildClusters = penalty_wildClusters
#         self.penalty_ante = penalty_ante

#         self.l = l
#         self.temp_h = temp_h
#         self.temp_c = temp_c

#         self.DEVICE = DEVICE

#         if p is None:
#             self.p = torch.zeros_like(s).to(DEVICE)
#         else:
#             self.p = p.to(DEVICE)

#         if p_nSpC is None:
#             # nSpC = s.shape[0] / self.n_clusters
#             # self.p_nSpC = lambda x : torch.log(-(x-(nSpC+1)))**(4)
#             self.p_nSpC = lambda x : x*0
#         else:
#             self.p_nSpC = p_nSpC

#         if init_h is None:
#             self.h = self._initialize_multihot_matrix(self.s.shape[0], self.n_clusters + self.n_wildClusters).to(self.DEVICE)
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

#         self.grad_clip = grad_clip

#         self.losses = {
#             'L_cs': [],
#             'L_nSpC': [],
#             'L_wildClusters': [],
#             'L_p': [],
#             'L_total': [],
#         }

#         self.i_iter = 0

#         # torch.autograd.set_detect_anomaly(True)


#     def _initialize_multihot_matrix(self, n_samples, n_clusters):
#         # h = torch.rand(size=(n_samples, n_clusters))  ## Random initialization
#         # h = h / h.sum(dim=0, keepdim=True)

#         h = torch.cat(
#             (
#                 torch.rand(n_samples,n_clusters) * 1, 

#                 # torch.rand(n_samples,n_clusters-1) * 20, 
#                 # (torch.ones(n_samples,1) + torch.rand(n_samples,1)*0.2) * 20
#             ),
#             dim=1,
#         )
#         return h
    
#     def _make_cluster_similarity_matrix(self, s, h, DEVICE='cpu'):
#         # h = torch.nn.functional.softmax(h/temp_h, dim=1)
#     #     return torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h))  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#     #     return (  torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h)) * torch.eye(h.shape[1]).to(DEVICE)  +  torch.einsum('ab, abcd -> cd', s**4, torch.einsum('ac, bd -> abcd', h,h)) * (torch.logical_not(torch.eye(h.shape[1]).to(DEVICE)))*0.05  )  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
#         return torch.einsum('ab, ac, bd -> cd', s, h, h)  /  \
#             ( (torch.eye(h.shape[1]).to(DEVICE) * self.ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * self.ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )

#     def fit(
#         self,
#         n_iter=200,
#         verbose=False,
#         verbose_interval=10,
#     ):

#         for _ in range(n_iter):
#             self.optimizer.zero_grad()

#             self.h_p = torch.nn.functional.softmax(self.h/self.temp_h, dim=1)
#             # self.h_p = torch.nn.functional.softplus(self.h/self.temp_h, beta=50, threshold=10)
#             self.c = self._make_cluster_similarity_matrix(
#                 s=self.s,
#                 h=self.h_p, 
#                 DEVICE=self.DEVICE
#             )
#             # print(self.h_p.sum(0))
#             # print(self.p_nSpC(self.h_p.sum(0)))
            
#             self.L_cs = (
#                 torch.nn.functional.cross_entropy(
#                     self.c[:-self.n_wildClusters, :-self.n_wildClusters]/self.temp_c, 
#                     torch.arange(self.n_clusters).to(self.DEVICE),
#                     reduction='none',
#                 ) @ (torch.tanh(self.h_p.sum(0))[:-self.n_wildClusters] ** self.penalty_ante)) / (self.c.shape[0]-self.n_wildClusters)
#                 # ) 
#             # )
#             # ).mean()
#             # ).sum() / (self.c.shape[0]-self.n_wildClusters)

#             self.p_cs = self._make_cluster_similarity_matrix(s=self.p, h=self.h_p, DEVICE=self.DEVICE)
#             self.L_p = torch.maximum(self.p_cs.diag()[:-self.n_wildClusters], torch.as_tensor([0], device=self.DEVICE)).mean()  ## I think there is some numerical instability that causes p_cs to have negative values even if p and h_p are positive

#             # print(self._make_cluster_similarity_matrix(s=self.p, h=self.h_p, DEVICE=self.DEVICE))
#             self.L_wildClusters = (self.h_p.mean(0)[-self.n_wildClusters:]*self.penalty_wildClusters).mean()
#             self.L_nSpC = self.p_nSpC(self.h_p.sum(0))[:-self.n_wildClusters].mean()
#             self.loss = self.L_cs + self.L_nSpC + self.L_p + self.L_wildClusters

#             if torch.isnan(self.loss):
#                 print(f'STOPPED on iter {self.i_iter}: loss is NaN')

#                 print('NaN detected in c') if torch.isnan(self.c).any() else None
#                 print('NaN detected in p') if torch.isnan(self.p).any() else None
#                 print('NaN detected in h') if torch.isnan(self.h).any() else None
#                 print('NaN detected in h_p') if torch.isnan(self.h_p).any() else None
#                 print('NaN detected in L_cs') if torch.isnan(self.L_cs).any() else None
#                 print('NaN detected in L_nSpC') if torch.isnan(self.L_nSpC).any() else None
#                 print('NaN detected in L_p') if torch.isnan(self.L_p).any() else None
#                 print('NaN detected in L_wildClusters') if torch.isnan(self.L_wildClusters).any() else None
#                 break

#             self.loss.backward()
#             if self.grad_clip is not None:
#                 torch.nn.utils.clip_grad_norm_(self.h, self.grad_clip)
#             self.optimizer.step()

#             self.losses['L_cs'].append(self.L_cs.item())
#             self.losses['L_nSpC'].append(self.L_nSpC.item())
#             self.losses['L_p'].append(self.L_p.item())
#             self.losses['L_wildClusters'].append(self.L_wildClusters.item())
#             self.losses['L_total'].append(self.loss.item())

#             if verbose and self.i_iter % verbose_interval == 0:
#                 print(f'iter: {self.i_iter}:  loss_total: {self.loss.item():.4f}   loss_cs: {self.L_cs.item():.4f}  loss_nSpC: {self.L_nSpC.item():.4f}  loss_p: {self.L_p.item():.4f}  loss_wildClusters: {self.L_wildClusters.item():.4f}')

#             # if self.loss.item() > 10:
#             #     break
#             self.i_iter += 1
            


#             # break



class Constrained_rich_clustering:
    def __init__(
        self,
        c,
        h,
        w=None,
        m_init=None,
        optimizer_partial=None,
        dmCEL_temp=1,
        dmCEL_sigSlope=5,
        dmCEL_sigCenter=0.5,
        sampleWeight_softplusKwargs={'beta': 20, 'threshold': 50},
        sampleWeight_penalty=1e2,
        fracWeighted_goalFrac=1.0,
        fracWeighted_sigSlope=1,
        fracWeighted_center=0.5,
        fraceWeight_penalty=1e3,
        tol_convergence=1e-2,
        window_convergence=100,
        freqCheck_convergence=100,
        verbose=True,
        freq_verbose=100,
    ):
        self._n_samples = c.shape[0]
        self._n_clusters = h.shape[1]

        self._DEVICE = c.device
        self.c = c
        self.h = h.to(self._DEVICE)
        self.w = w if w is not None else torch.ones(self._n_samples).type(torch.float32).to(self._DEVICE)
        self.m = m_init if m_init is not None else (torch.ones(self._n_clusters)*0.1 + torch.rand(self._n_clusters)*0.05).type(torch.float32).to(self._DEVICE)
        self.m.requires_grad=True

        self._optimizer = optimizer_partial(params=[self.m]) if optimizer_partial is not None else torch.optim.Adam(params=[self.m], lr=1e-2, betas=(0.9, 0.900))
        
        self._sampleWeight_penalty = sampleWeight_penalty
        self._fracWeight_penalty = fraceWeight_penalty

        self._dmCEL = self._DoubleMasked_CEL(
            n_clusters=self._n_clusters,
            device=self._DEVICE,
            temp=dmCEL_temp,
            sig_slope=dmCEL_sigSlope,
            sig_center=dmCEL_sigCenter,
        )

        self._loss_fracWeighted = self._Loss_fracWeighted(
            h=self.h,
            w=self.w,
            goal_frac=fracWeighted_goalFrac,
            sig_slope=fracWeighted_sigSlope,
            sig_center=fracWeighted_center,
        )

        self._loss_sampleWeight = self._Loss_sampleWeight(
            softplus_kwargs=sampleWeight_softplusKwargs,
        )

        self._convergence_checker = self._Convergence_checker(
            tol_convergence=tol_convergence,
            window_convergence=window_convergence,
        )

        self._tol_convergence = tol_convergence
        self._window_convergence = window_convergence
        self._freqCheck_convergence = freqCheck_convergence
        self._verbose = verbose
        self._freq_verbose = freq_verbose

        self._i_iter = 0
        self.losses_logger = {
            'loss': [],
            'L_cs': [],
            'L_fracWeighted': [],
            'L_sampleWeight': [],
        }

    def fit(
        self, 
        max_iter=1e4,
        verbose=True, 
        verbose_interval=100
    ):
        loss_smooth = np.nan
        diff_window_convergence = np.nan
        while self._i_iter <= max_iter:
            self._optimizer.zero_grad()

            L_cs = self._dmCEL(c=self.c, m=self.m)  ## 'cluster similarity loss'
            L_sampleWeight = self._loss_sampleWeight(self.h, self.activate_m()) * self._sampleWeight_penalty
            L_fracWeighted = self._loss_fracWeighted(self.activate_m()) * self._fracWeight_penalty

            self._loss = L_cs + L_fracWeighted + L_sampleWeight

            if torch.isnan(self._loss):
                print(f'STOPPING EARLY: loss is NaN. iter: {self._i_iter}  loss: {self._loss.item():.4f}  L_cs: {L_cs.item():.4f}  L_fracWeighted: {L_fracWeighted.item():.4f}  L_sampleWeight: {L_sampleWeight.item():.4f}')
                break

            self._loss.backward()
            self._optimizer.step()

            self.losses_logger['loss'].append(self._loss.item())
            self.losses_logger['L_cs'].append(L_cs.item())
            self.losses_logger['L_fracWeighted'].append(L_fracWeighted.item())
            self.losses_logger['L_sampleWeight'].append(L_sampleWeight.item())

            if self._i_iter%self._freqCheck_convergence==0 and self._i_iter>self._window_convergence:
                diff_window_convergence, loss_smooth, converged = self._convergence_checker(self.losses_logger['loss'])
                if converged:
                    print(f"STOPPING: Convergence reached in {self._i_iter} iterations.  loss: {self.losses_logger['loss'][-1]:.4f}  loss_smooth: {loss_smooth:.4f}")
                    break

            if verbose and self._i_iter % verbose_interval == 0:
                print(f'iter: {self._i_iter}:  loss_total: {self._loss.item():.4f}   loss_cs: {L_cs.item():.4f}  loss_fracWeighted: {L_fracWeighted.item():.4f}  loss_sampleWeight: {L_sampleWeight.item():.4f}  diff_loss: {diff_window_convergence:.4f}  loss_smooth: {loss_smooth:.4f}')

            self._i_iter += 1


    def predict(
        self,
        m_threshold=0.5
    ):   
        m_bool = self.activate_m() > m_threshold
        h_preds = (self.h[:, m_bool] * (torch.arange(m_bool.sum(), device=self._DEVICE)[None,:]+1)).detach().cpu()
        h_preds[h_preds==0] = -1

        preds = torch.max(h_preds, dim=1)[0]
        preds[torch.isinf(preds)] = -1
        
        h_m = (self.h * self.activate_m()[None,:]).detach().cpu()
        confidence = h_m.var(1) / h_m.mean(1)
        
        self.preds = preds
        self.confidence = confidence
        
        return preds, confidence

    def activate_m(self):
        return self._dmCEL.activation(self.m)


    class _DoubleMasked_CEL:
        def __init__(
            self,
            n_clusters,
            device='cpu',
            temp=1,
            sig_slope=5,
            sig_center=0.5,
        ):
            self.labels = torch.arange(n_clusters, device=device, dtype=torch.int64)
            self.CEL = torch.nn.CrossEntropyLoss(reduction='none')
            self.temp = temp
            self.activation = self.make_sigmoid_function(sig_slope, sig_center)
            
        def make_sigmoid_function(
            self,
            sig_slope=5,
            sig_center=0.5,
        ):
            return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))
            
        def __call__(self, c, m):
            mp = self.activation(m)  ## constrain to be 0-1
            cm = c * mp[None,:]  ## 'c masked'. Mask only applied to columns.
            lv = self.CEL(cm/self.temp, self.labels)  ## 'loss vector' showing loss of each row (each cluster)
            l = lv @ mp  ## 'loss'
            return l

    class _Loss_fracWeighted:
        def __init__(
            self,
            h,
            w,
            goal_frac=1.0,
            sig_slope=5,
            sig_center=0.5,
        ):
            self.h_w = h.type(torch.float32) * w[None,:]
        
            self.goal_frac = goal_frac
            self.sigmoid = self.make_sigmoid_function(sig_slope, sig_center)
            
        def make_sigmoid_function(
            self,
            sig_slope=5,
            sig_center=0.5,
        ):
            return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))

        def generate_sampleWeights(self, m):
            return self.h_w @ m
        def activate_sampleWeights(self, sampleWeights):
            return self.sigmoid(sampleWeights)

        def __call__(self, m):
            return (self.activate_sampleWeights(self.generate_sampleWeights(m)) - self.goal_frac).mean()**2

    class _Loss_sampleWeight:
        def __init__(
            self,
            softplus_kwargs=None,
        ):
            self.softplus = torch.nn.Softplus(**softplus_kwargs)
            self.weight_high = 1
            self.offset = self.activate_sampleWeights(torch.ones(1, dtype=torch.float32)).max()
        
        def generate_sampleWeights(self, h, m):
            return h @ m
        def activate_sampleWeights(self, sampleWeights):
            return self.softplus(sampleWeights - 1 - 4/self.softplus.beta)*self.weight_high

        def __call__(self, h, m):
            return (self.activate_sampleWeights(self.generate_sampleWeights(h, m))).mean()

    class _Convergence_checker:
        def __init__(
            self,
            tol_convergence=1e-2,
            window_convergence=100,
        ):
            self.window_convergence = window_convergence
            self.tol_convergence = tol_convergence

            self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

        def OLS(self, y):
            X = self.line_regressor
            theta = torch.inverse(X.T @ X) @ X.T @ y
            y_rec = X @ theta
            bias = theta[-1]
            theta = theta[:-1]

            return theta, y_rec, bias

        def __call__(
            self,
            loss_history,
        ):
            loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
            theta, y_rec, bias = self.OLS(y=loss_window)

            diff_window_convergence = (y_rec[-1] - y_rec[0])
            loss_smooth = loss_window.mean()
            converged = True if torch.abs(diff_window_convergence) < self.tol_convergence else False
            return diff_window_convergence.item(), loss_smooth.item(), converged
            

    def plot_loss(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.losses_logger['loss'], linewidth=4)
        plt.plot(self.losses_logger['L_cs'])
        plt.plot(self.losses_logger['L_fracWeighted'])
        plt.plot(self.losses_logger['L_sampleWeight'])

        plt.legend(self.losses_logger.keys())
        plt.xlabel('iteration')
        plt.ylabel('loss')

    def plot_clusterWeights(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(self.activate_m().detach().cpu().numpy(), bins=50)
        plt.xlabel('cluster weight')
        plt.ylabel('count')

    def plot_sampleWeights(self):
        import matplotlib.pyplot as plt

        sampleWeights = self._loss_sampleWeight.generate_sampleWeights(self.h, self.activate_m()).detach().cpu().numpy()

        plt.figure()
        plt.hist(sampleWeights, bins=50)
        plt.xlabel('sample weight')
        plt.ylabel('count')

    def plot_labelCounts(self):
        import matplotlib.pyplot as plt

        if hasattr(self, 'preds'):
            preds = self.preds
        else:
            preds, confidence = self.predict()
            
        labels_unique, label_counts = np.unique(preds, return_counts=True)

        plt.figure()
        plt.bar(labels_unique, label_counts)
        plt.xlabel('label')
        plt.ylabel('count')