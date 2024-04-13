from typing import Tuple, Optional, Union
import math
import gc
import copy

import sklearn.decomposition
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

# import cuml
# import cuml.decomposition
# import cupy


###########################
########## PCA ############
###########################

def simple_pca(X , n_components=None , mean_sub=True, zscore=False, plot_pref=False , n_PCs_toPlot=2):
    """
    Performs PCA on X.
    RH 2021

    Args:
        X (np.ndarray):
            Data to be decomposed.
            2-D array. Columns are features, rows are samples.
        n_components (int):
            Number of components to keep. If None, then
             n_components = X.shape[1]
        mean_sub (bool):
            Whether or not to mean subtract ('center') the
             columns.
        zscore (bool):
            Whether or not to z-score the columns. This is
             equivalent to doing PCA on the correlation-matrix.
        plot_pref (bool):
            Whether or not to plot the first n_PCs_toPlot of the
             PCA.
        n_PCs_toPlot (int):
            Number of PCs to plot.

    Returns:
        components (np.ndarray):
            The components of the decomposition.
            2-D array.
            Each column is a component vector. Each row is a
             feature weight.
        scores (np.ndarray):
            The scores of the decomposition.
            2-D array.
            Each column is a score vector. Each row is a
             sample weight.
        EVR (np.ndarray):
            The explained variance ratio of each component.
            1-D array.
            Each element is the explained variance ratio of
             the corresponding component.
    """
    if mean_sub and not zscore:
        X = X - np.mean(X, axis=0)
    if zscore:
        # X = scipy.stats.zscore(X, axis=0)
        X = X - np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = X / stds[None,:]
    
    if n_components is None:
        n_components = X.shape[1]
    decomp = sklearn.decomposition.PCA(n_components=n_components)
    decomp.fit_transform(X)
    components = decomp.components_
    scores = decomp.transform(X)
    
    if plot_pref:
        fig , axs = plt.subplots(4 , figsize=(7,15))
        axs[0].plot(np.arange(n_components)+1,
                    decomp.explained_variance_ratio_)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('component #')
        axs[0].set_ylabel('explained variance ratio')

        axs[1].plot(np.arange(n_components)+1, 
                    np.cumsum(decomp.explained_variance_ratio_))
        axs[1].set_xscale('log')
        axs[1].set_ylabel('cumulative explained variance ratio')

        axs[2].plot(scores[:,:n_PCs_toPlot])
        axs[2].set_xlabel('sample num')
        axs[2].set_ylabel('a.u.')

        axs[3].plot(components.T[:,:n_PCs_toPlot])
        axs[3].set_xlabel('feature num')
        axs[3].set_ylabel('score')
    
    return components , scores , decomp.explained_variance_ratio_


def svd_flip(
    u: torch.Tensor, 
    v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sign correction to ensure deterministic output from SVD.

    The output from SVD does not have a unique sign. This function corrects the
    sign of the output to ensure deterministic output from the SVD function.

    RH 2024

    Args:
        u (torch.Tensor):
            The left singular vectors.
        v (torch.Tensor):
            The right singular vectors.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            u (torch.Tensor):
                The corrected left singular vectors.
            v (torch.Tensor):
                The corrected right singular vectors.
    """
    as_tensor = lambda x: torch.as_tensor(x) if isinstance(x, np.ndarray) else x
    u, v = (as_tensor(var) for var in (u, v))
    
    max_abs_cols = torch.argmax(torch.abs(u), dim=0)
    signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs.unsqueeze(-1)
    return u, v


class PCA(torch.nn.Module, sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Principal Component Analysis (PCA) module.

    This module performs PCA on the input data and returns the principal
    components and the explained variance. The PCA is performed using the
    singular value decomposition (SVD) method. This class follows sklearn's PCA
    implementation and style and is subclassed from torch.nn.Module,
    sklearn.base.BaseEstimator, and sklearn.base.TransformerMixin. The
    decomposed variables (components, explained_variance, etc.) are stored as
    buffers so that they are stored in the state_dict, respond to .to() and
    .cuda(), and are saved when the model is saved.

    RH 2024

    Args:
        n_components (Optional[int]):
            Number of principal components to retain. If ``None``, all
            components are retained.
        center (bool):
            If ``True``, the data is mean-subtracted before performing SVD.
        zscale (bool):
            If ``True``, the data is z-scored before performing SVD. Equivalent
            of doing eigenvalue decomposition on the correlation matrix.
        whiten (bool):
            If ``True``, the principal components are divided by the square root
            of the explained variance.
        use_lowRank (bool):
            If ``True``, the low-rank SVD is used. This is faster but less
            accurate. Uses torch.svd_lowrank instead of torch.linalg.svd.
        lowRank_niter (int):
            Number of subspace iterations for low-rank SVD. See
            torch.svd_lowrank for more details.

    Attributes:
        n_components (int):
            Number of principal components to retain.
        whiten (bool):
            If ``True``, the principal components are divided by the square root
            of the explained variance.
        device (str):
            The device where the tensors will be stored.
        dtype (torch.dtype):
            The data type to use for the tensor.
        components (torch.Tensor):
            The principal components.
        explained_variance (torch.Tensor):
            The explained variance.
        explained_variance_ratio (torch.Tensor):
            The explained variance ratio.

    Example:
        .. highlight:: python
        .. code-block:: python

            X = torch.randn(100, 10)
            pca = PCA(n_components=5)
            pca.fit(X)
            X_pca = pca.transform(X)
    """
    def __init__(
        self,
        n_components: Optional[int] = None,
        center: bool = True,
        zscale: bool = False,
        whiten: bool = False,
        use_lowRank: bool = False,
        lowRank_niter: int = 2,
    ):
        """
        Initializes the PCA module with the provided parameters.
        """
        super(PCA, self).__init__()
        self.n_components = n_components
        self.center = center
        self.zscale = zscale
        self.whiten = whiten
        self.use_lowRank = use_lowRank
        self.lowRank_niter = lowRank_niter

    def prepare_input(
        self,
        X: torch.Tensor,
        center: bool,
        zscale: bool
    ) -> torch.Tensor:
        """
        Prepares the input data for PCA.

        Args:
            X (torch.Tensor):
                The input data to prepare.
            center (bool):
                If ``True``, the data is mean-subtracted.
            zscale (bool):
                If ``True``, the data is z-scored.

        Returns:
            (torch.Tensor):
                The prepared input data.
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X)
        assert isinstance(X, torch.Tensor), 'Input must be a torch.Tensor.'
        X = X[:, None] if X.ndim == 1 else X
        assert X.ndim == 2, 'Input must be 2D.'

        if center:
            mean_ = torch.mean(X, dim=0)
            X = X - mean_
            self.register_buffer('mean_', mean_)
        if zscale:
            std_ = torch.std(X, dim=0)
            X = X / std_
            self.register_buffer('std_', std_)
        return X

    def fit(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fits the PCA module to the input data.

        Args:
            X (torch.Tensor):
                The input data to fit the PCA module to. Should be shape
                (n_samples, n_features).

        Returns:
            self (PCA object):
                Returns the PCA object.
        """
        self._fit(X)
        return self

    def _fit(
        self,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fits the PCA module to the input data.

        Args:
            X (torch.Tensor):
                The input data to fit the PCA module to. Should be shape
                (n_samples, n_features).

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                U (torch.Tensor):
                    The left singular vectors. Shape (n_samples, n_components).
                S (torch.Tensor):
                    The singular values. Shape (n_components,).
                V (torch.Tensor):
                    The right singular vectors. Shape (n_features,
                    n_components).
        """
        self.n_samples_, self.n_features_ = X.shape
        self.n_components_ = min(self.n_components, self.n_features_) if self.n_components is not None else self.n_features_

        X = self.prepare_input(X, center=self.center, zscale=self.zscale)
        if self.use_lowRank:
            U, S, Vh = torch.svd_lowrank(X, q=self.n_components_, niter=self.lowRank_niter)
            Vh = Vh.T  ## torch.svd_lowrank returns Vh transposed.
        else:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)  ## U: (n_samples, n_features), S: (n_features,), Vh: (n_features, n_features). Vh is already transposed.
        U, Vh = svd_flip(U, Vh)

        explained_variance_ = S**2 / (self.n_samples_ - 1)
        explained_variance_ratio_ = explained_variance_ / torch.sum(explained_variance_)

        components_ = Vh[:self.n_components_]
        singular_values_ = S[:self.n_components_]
        explained_variance_ = explained_variance_[:self.n_components_]
        explained_variance_ratio_ = explained_variance_ratio_[:self.n_components_]

        [self.register_buffer(name, value) for name, value in zip(
            ['components_', 'singular_values_', 'explained_variance_', 'explained_variance_ratio_'],
            [components_, singular_values_, explained_variance_, explained_variance_ratio_]
        )]

        return U, S, Vh
    
    def transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transforms the input data using the fitted PCA module.

        Args:
            X (torch.Tensor):
                The input data to transform.
            y (Optional[torch.Tensor]):
                Ignored. This parameter exists to match the sklearn API.

        Returns:
            (torch.Tensor):
                The transformed data. Will be shape (n_samples, n_components).
        """
        assert hasattr(self, 'components_'), 'PCA module must be fitted before transforming data.'
        X = self.prepare_input(X, center=self.center, zscale=self.zscale)
        X_transformed = X @ self.components_.T
        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_)
        return X_transformed
    
    def fit_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Fits the PCA module to the input data and transforms the input data.

        Args:
            X (torch.Tensor):
                The input data to fit the PCA module to and transform.

        Returns:
            (torch.Tensor):
                The transformed data.
        """
        self.n_samples_, self.n_features_ = X.shape

        U, S, V = self._fit(X)
        U = U[:, :self.n_components_]
        
        if self.whiten:
            U *= math.sqrt(self.n_samples_ - 1)
        else:
            U *= S[:self.n_components_]

        return U
    
    def inverse_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transforms the input data using the fitted PCA module.

        Args:
            X (torch.Tensor):
                The input data to inverse transform. Should be shape (n_samples,
                n_components).

        Returns:
            (torch.Tensor):
                The inverse transformed data. Will be shape (n_samples,
                n_features).
        """
        assert hasattr(self, 'components_'), 'PCA module must be fitted before transforming data.'
        X = self.prepare_input(X, center=False, zscale=False)
        
        if self.whiten:
            scaled_components = torch.sqrt(self.explained_variance_)[:, None] * self.components_
        else:
            scaled_components = self.components_
        
        X = X @ scaled_components

        if self.zscale:
            assert hasattr(self, 'std_'), 'self.zscale is True, but std_ is not found.'
            X = X * self.std_
        if self.center:
            assert hasattr(self, 'mean_'), 'self.center is True, but mean_ is not found.'
            X = X + self.mean_

        return X


def torch_pca(
    X_in, 
    device='cpu', 
    mean_sub=True, 
    zscore=False, 
    rank=None, 
    return_cpu=True, 
    return_numpy=False,
    cuda_empty_cache=True,
):
    """
    Principal Components Analysis for PyTorch.
    If using GPU, then call torch.cuda.empty_cache() after.
    RH 2021

    Args:
        X_in (torch.Tensor or np.ndarray):
            Data to be decomposed.
            2-D array. Columns are features, rows are samples.
            PCA will be performed column-wise.
        device (str):
            Device to use. ie 'cuda' or 'cpu'. Use a function 
             torch_helpers.set_device() to get.
        mean_sub (bool):
            Whether or not to mean subtract ('center') the 
             columns.
        zscore (bool):
            Whether or not to z-score the columns. This is 
             equivalent to doing PCA on the correlation-matrix.
        rank (int):
            Maximum estimated rank of decomposition. If None,
             then rank is X.shape[1]
        return_cpu (bool):  
            Whether or not to force returns/outputs to be on 
             the 'cpu' device. If False, and device!='cpu',
             then returns will be on device.
        return_numpy (bool):
            Whether or not to force returns/outputs to be
             numpy.ndarray type.
        cuda_empty_cache (bool):
            Whether or not to call torch.cuda.empty_cache()
            Only relevant if device!='cpu'.

    Returns:
        components (torch.Tensor or np.ndarray):
            The components of the decomposition. 
            2-D array.
            Each column is a component vector. Each row is a 
             feature weight.
        scores (torch.Tensor or np.ndarray):
            The scores of the decomposition.
            2-D array.
            Each column is a score vector. Each row is a 
             sample weight.
        singVals (torch.Tensor or np.ndarray):
            The singular values of the decomposition.
            1-D array.
            Each element is a singular value.
        EVR (torch.Tensor or np.ndarray):
            The explained variance ratio of each component.
            1-D array.
            Each element is the explained variance ratio of
             the corresponding component.
    """
    
    if isinstance(X_in, torch.Tensor) == False:
        X = torch.from_numpy(X_in).to(device)
    elif str(X_in.device) != device:
        X = X_in.to(device)
    else:
        X = X_in
        
    if mean_sub and not zscore:
        X = X - torch.mean(X, dim=0)
    if zscore:
        X = X - torch.mean(X, dim=0)
        stds = torch.std(X, dim=0)
        idx_zeroStd = torch.where(stds==0)[0]
        stds[idx_zeroStd] = 1.0
        X = X / stds
        
    if rank is None:
        rank = min(list(X.shape))
    
    (U,S,V) = torch.pca_lowrank(X, q=rank, center=False, niter=2)
    components = V
    scores = torch.matmul(X, V[:, :rank])

    singVals = (S**2)/(len(S)-1)
    EVR = (singVals) / torch.sum(singVals)
    
    if return_cpu:
        components = components.cpu()
        scores = scores.cpu()
        singVals = singVals.cpu()
        EVR = EVR.cpu()
    if return_numpy:
        components = components.cpu().numpy()
        scores = scores.cpu().numpy()
        singVals = singVals.cpu().numpy()
        EVR = EVR.cpu().numpy()

    if cuda_empty_cache and device!='cpu':
        torch.cuda.empty_cache()
        gc.collect()
        
    return components, scores, singVals, EVR

def unmix_pcs(pca_components, weight_vecs):
    """
    Transforms weight_vecs into pca_components space
    RH 2021
    """
    if weight_vecs.ndim == 1:
        weight_vecs = weight_vecs[:,None]
    if type(pca_components) == np.ndarray:
        zeros = np.zeros
    elif type(pca_components) == torch.Tensor:
        zeros = torch.zeros
    
    mixing_vecs = zeros((pca_components.shape[1], weight_vecs.shape[1]))
    mixing_vecs[:weight_vecs.shape[0],:] = weight_vecs

    return pca_components @ mixing_vecs


def dimensionality_pca(
    X, 
    ev=[0.10,0.25,0.50,0.90,0.95], 
    mean_sub=True, 
    zscore=False, 
    device='cpu', 
    kwargs_torchPCA={}
):
    """
    Returns the number of components needed to explain a certain 
     amount of variance. 
    Calculates the 'linear embedding dimensionality'.
    Uses torch_pca() function to perform PCA.
    RH 2022

    Args:
        X (np.ndarray or torch.Tensor):
            Data to be decomposed.
            2-D array. Columns are features, rows are samples.
        ev (list or np.ndarray):
            List of explained variance ratios to calculate the
             number of components needed to explain.
            ie [0.10,0.25,0.50,0.90,0.95]
        mean_sub (bool):
            Whether or not to mean subtract ('center') the
             columns.
        zscore (bool):
            Whether or not to z-score the columns. This is
             equivalent to doing PCA on the correlation-matrix.
        device (str):
            Device to use. ie 'cuda' or 'cpu'. Use a function
             torch_helpers.set_device() to get.

    Returns:
        n_components (np.ndarray):
            Number of components needed to explain the
             corresponding ev.
            Same array type (torch.Tensor or np.array) as X.
    """
    ## make sure ev is a list or np.ndarray
    ev = [ev] if isinstance(ev, (int,float)) else ev
    assert isinstance(ev, (list,np.ndarray)), 'RH ERROR: ev must be a list or np.ndarray.'
    ev = np.array(ev)

    ## assert there are no nans
    assert np.any(np.isnan(X)) == False, 'RH ERROR: X must not contain NaNs.'

    comps, scores, sv, EVR = torch_pca(
        X_in=X, 
        device=device,
        return_numpy=True,
        mean_sub=mean_sub,
        zscore=zscore,
        **kwargs_torchPCA,
    )
    EVR_cumsum = np.cumsum(np.concatenate([[0], EVR]))
    interp = scipy.interpolate.interp1d(x=EVR_cumsum, y=np.arange(0, len(EVR_cumsum)), kind='linear')
    return interp(ev)


def ZCA_whiten(
    X: Union[np.ndarray, torch.Tensor],
    V: Union[np.ndarray, torch.Tensor],
    S: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
):
    """
    ZCA whitening of data.
    See: https://jermwatt.github.io/control-notes/posts/zca_sphereing/ZCA_Sphereing.html
    RH 2024

    Args:
        X (np.ndarray or torch.Tensor):
            Data to be whitened. \n
            Shape (n_samples, n_features).
        V (np.ndarray or torch.Tensor):
            The principal components / eigenvectors. \n
            You can use PCA.components_ from sklearn.decomposition.PCA or
            PCA.components_ above. \n
            Shape (n_features, n_components).
        S (np.ndarray or torch.Tensor):
            The singular values / eigenvalues. \n
            You can use PCA.singular_values_ from sklearn.decomposition.PCA or
            PCA.singular_values_ above. \n
            Shape (n_components,).
        eps (float):
            Small value to prevent division by zero.

    Returns:
        X_zca (np.ndarray or torch.Tensor):
            The ZCA whitened data. \n
            Shape (n_samples, n_features).

    Demo:
        ..code-block:: python
        
            X = np.random.randn(100, 10)
            pca = PCA(n_components=5)
            pca.fit(X)
            X_zca = ZCA_whiten(
                X=X,
                V=pca.components_,
                S=pca.singular_values_,
                eps=1e-5,
            )
    """
    if isinstance(X, np.ndarray):
        mean, sqrt, diag = np.mean, np.sqrt, np.diag
    elif isinstance(X, torch.Tensor):
        mean, sqrt, diag = torch.mean, torch.sqrt, torch.diag

    X = X - mean(X, axis=0, keepdims=True)

    D_inv = diag(1.0 / (S + eps))
    W_zca = V.T @ D_inv @ V
    X_zca = X @ W_zca

    return X_zca


#######################################
########## Incremental PCA ############
#######################################

class IPCA_Dataset(Dataset):
    """
    see incremental_pca for demo
    """
    def __init__(self, 
                 X, 
                 mean_sub=True,
                 zscore=False,
                 preprocess_sample_method='random',
                 preprocess_sample_num=100,
                 device='cpu',
                 dtype=torch.float32):
        """
        Make a basic dataset.
        RH 2021

        Args:
            X (torch.Tensor or np.array):
                Data to make dataset from.
                2-D array. Columns are features, rows are samples.
            mean_sub (bool):
                Whether or not to mean subtract ('center') the
                 columns.
            zscore (bool):
                Whether or not to z-score the columns. This is
                 equivalent to doing PCA on the correlation-matrix.
            preprocess_sample_method (str):
                Method to use for sampling for mean_sub and zscore.
                'random' - uses random samples (rows) from X.
                'first' - uses the first rows of X.
            preprocess_sample_num (int):
                Number of samples to use for mean_sub and zscore.
            device (str):
                Device to use.
            dtype (torch.dtype):
                Data type to use.
        """
        # Upgrade here for using an on the fly dataset.
        self.X = torch.as_tensor(X, dtype=dtype, device=device) # first (0th) dim will be subsampled from
        
        self.n_samples = self.X.shape[0]

        self.mean_sub = mean_sub
        self.zscore = zscore

        if mean_sub or zscore:
            if preprocess_sample_method == 'random':
                self.preprocess_inds = torch.randperm(preprocess_sample_num)[:preprocess_sample_num]
            elif preprocess_sample_method == 'first':
                self.preprocess_inds = torch.arange(preprocess_sample_num)
            else:
                raise ValueError('preprocess_sample_method must be "random" or "first"')
            self.preprocess_inds = self.preprocess_inds.to(device)

        # Upgrade here for using an on the fly dataset.
        if mean_sub:
            self.mean_vals = torch.mean(self.X[self.preprocess_inds,:], dim=0)
        if zscore:
            self.std_vals = torch.std(self.X[self.preprocess_inds,:], dim=0)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns a single sample.

        Args:
            idx (int):
                Index of sample to return.
        """
        if self.mean_sub or self.zscore:
            out = self.X[idx,:] - self.mean_vals
        else:
            out = self.X[idx,:]

        if self.zscore:
            out = out / self.std_vals
            
        return out, idx

def incremental_pca(dataloader,
                    method='sklearn',
                    method_kwargs=None,
                    return_cpu=True,
                    ):
    """
    Incremental PCA using either sklearn or cuml.
    Keep batches small-ish to remain performat (~64 samples).
    RH 2021

    Args:
        dataloader (torch.utils.data.DataLoader):
            Data to be decomposed.
        method (str):
            Method to use.
            'sklearn' : sklearn.decomposition.PCA
            'cuml' : cuml.decomposition.IncrementalPCA
        method_kwargs (dict):
            Keyword arguments to pass to method.
            See method documentation for details.
        device (str):
            Device to use.
            Only used if method is 'cuml'
        return_cpu (bool):  
            Whether or not to force returns/outputs to be on 
             the 'cpu' device. If False, and device!='cpu',
             then returns will be on device.
        return_numpy (bool):
            Whether or not to force returns/outputs to be
             numpy.ndarray type.

    Returns:
        components (torch.Tensor or np.ndarray):
            The components of the decomposition. 
            2-D array.
            Each column is a component vector. Each row is a 
             feature weight.
        EVR (torch.Tensor or np.ndarray):
            The explained variance ratio of each component.
            1-D array.
            Each element is the explained variance ratio of
             the corresponding component.
        object_params (dict):
            Dictionary of parameters used to create the
             decomposition.

    demo:
    import torch_helpers, cupy_helpers

    cupy_helpers.set_device() # calls: cupy.cuda.Device(DEVICE_NUM).use()

    dataset = decomposition.IPCA_Dataset(   X, 
                                            mean_sub=True,
                                            zscore=False,
                                            preprocess_sample_method='random',
                                            preprocess_sample_num=10000,
                                            device='cpu',
                                            dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(   dataset, 
                                                batch_size=64, 
                                                drop_last=True,
                                                shuffle=False, 
                                                num_workers=0, 
                                                pin_memory=False)

    cuml_kwargs = {
                "handle": cuml.Handle(),
                "n_components": 20,
                "whiten": False,
                "copy": False,
                "batch_size": None,
                "verbose": True,
                "output_type": None
    }

    sk_kwargs = {
                    "n_components": 20,
                    "whiten": False,
                    "copy": False,
                    "batch_size": None,
    }

    components, EVR, ipca = decomposition.incremental_pca(dataloader,
                                    method='cuml',
                                    method_kwargs=cuml_kwargs,
                                    return_cpu=True)
    scores = decomposition.ipca_transform(dataloader, components)
    """
    
    if method_kwargs is None:
        method_kwargs = {}

    if method == 'sklearn':
        ipca = sklearn.decomposition.IncrementalPCA(**method_kwargs)
    elif method == 'cuml':
        ipca =    cuml.decomposition.IncrementalPCA(**method_kwargs)
    
    for iter_batch, batch in enumerate(tqdm(dataloader)):
        if method == 'sklearn':
            batch = batch[0].cpu().numpy()
        if method == 'cuml':
            batch = cupy.asarray(batch[0])
        ipca.partial_fit(batch)
    
    if (return_cpu) and (method=='cuml'):
        components = ipca.components_.get()
    else:
        components = ipca.components_

    EVR = ipca.explained_variance_ratio_

    return components, EVR, ipca

def ipca_transform(dataloader, components):
    """
    Transform data using incremental PCA.
    RH 2020

    Args:
        dataloader (torch.utils.data.DataLoader):
            Data to be decomposed.
        components (torch.Tensor or np.ndarray):
            The components of the decomposition. 
            2-D array.
            Each column is a component vector. Each row is a 
             feature weight.
    """
    out = []
    for iter_batch, batch in enumerate(dataloader):
        out.append(batch[0] @ components.T)
    return torch.cat(out, dim=0)