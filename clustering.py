import sklearn
import numpy as np

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