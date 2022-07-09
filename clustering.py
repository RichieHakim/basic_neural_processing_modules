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


class rich_clust():
    """
    Basic clustering algorithm based on gradient descent.
    The approach is to optimize a 'multihot matrix' (h),
     which is an n_samples x n_clusters matrix describing
     the membership of each sample to each cluster. This matrix
     is used to mask a similarity matrix (s); allowing for the
     calculation of the mean pairwise similarity within each 
     cluster and between each pair of clusters. The loss
     function maximizes the within similarity and minimizes the
     between similarity.
    Functionally, the result ends up similar to k-means
     clustering.
    This method is not great on its own, but is useful because
     it allows for custom penalties to be applied to the loss.
    It struggles due to a non-convex solution space which
     results in sensitivity to initial conditions and weird 
     sensitivities to hyperparameters.
    RH 2022
    """

    def __init__(
        self,
        s,
        n_clusters=2,
        l=1,
        temp_h=1,
        temp_c=1,
        optimizer=None,
        DEVICE='cpu',
        init_h=None,
    ):
        """
        Args:
            s (torch.Tensor):
                Similarity matrix.
                shape: (n_samples, n_samples)
                dtype: torch.float32
                The diagonal should be zeros.
                Example:
                    d = torch.cdist(data, data, p=2).type(torch.float32)
                    s = torch.maximum(1-d, torch.as_tensor([0]))
                    s = s * torch.logical_not(torch.eye(s.shape[0]))
            n_clusters (int):
                The number of clusters to find.
            l (float):
                'Locality' parameter. The exponent applied to 
                  the similarity matrix. Higher values allow for
                  more non-convex clusters, but can result in 
                  instability and cluster-splitting.
            temp_h (float):
                The temperature for the multihot matrix. Higher
                 values result in fuzzier cluster edges.
            temp_c (float):
                The temperature for the cluster membership matrix.
                Higher values result in globally less confident
                 cluster scores.
            optimizer (partial torch.optim.Optimizer):
                Optional. If None, then Adam is used.
                The partial optimizer to use.
                Can be constructed like: 
                    functools.partial(torch.optim.Adam, lr=0.01, weight_decay=0.00001)
            DEVICE (str):
                The device to use. Default is 'cpu'.
            init_h (torch.Tensor):
                The initial multihot matrix. If None, then
                 random initialization is used.
        """
    
        self.s = s.to(DEVICE)**l
        self.n_clusters = n_clusters

        self.l = l
        self.temp_h = temp_h
        self.temp_c = temp_c

        self.DEVICE = DEVICE

        if init_h is None:
            self.h = self._initialize_multihot_matrix(self.s.shape[0], self.n_clusters).to(self.DEVICE)
        else:
            self.h = init_h.to(self.DEVICE)
        self.h.requires_grad = True

        self.ii_normFactor = lambda i   : i * (i-1)
        self.ij_normFactor = lambda i,j : i * j

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                [self.h], 
                lr=0.1,
                # weight_decay=1*10**-6
            )
        else:
            self.optimizer = optimizer(params=[self.h])


    def _initialize_multihot_matrix(self, n_samples, n_clusters):
        h = torch.rand(size=(n_samples, n_clusters))  ## Random initialization
        return h
    
    def _make_cluster_similarity_matrix(self, s, h, temp_h, DEVICE='cpu'):
        h = torch.nn.functional.softmax(h/temp_h, dim=1)
    #     return torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h))  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
    #     return (  torch.einsum('ab, abcd -> cd', s**1, torch.einsum('ac, bd -> abcd', h,h)) * torch.eye(h.shape[1]).to(DEVICE)  +  torch.einsum('ab, abcd -> cd', s**4, torch.einsum('ac, bd -> abcd', h,h)) * (torch.logical_not(torch.eye(h.shape[1]).to(DEVICE)))*0.05  )  /  ( (torch.eye(h.shape[1]).to(DEVICE) * ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )
        return torch.einsum('ab, ac, bd -> cd', s, h, h)  /  \
            ( (torch.eye(h.shape[1]).to(DEVICE) * self.ii_normFactor(h.sum(0))) + ((1-torch.eye(h.shape[1]).to(DEVICE)) * self.ij_normFactor(*torch.meshgrid((h.sum(0), h.sum(0)), indexing='ij'))) )

    def fit(self, n_iter=200):

        for i_iter in range(n_iter):
            self.optimizer.zero_grad()

            self.c = self._make_cluster_similarity_matrix(
                s=self.s,
                h=self.h, 
                temp_h=self.temp_h,
                DEVICE=self.DEVICE
            )
            
            self.L_cs = torch.nn.functional.cross_entropy(self.c/self.temp_c, torch.arange(self.n_clusters).to(self.DEVICE))
            self.loss = self.L_cs
            self.loss.backward()
            self.optimizer.step()
