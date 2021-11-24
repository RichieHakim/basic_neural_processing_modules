import sklearn.decomposition
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import torch
import gc

def simple_pca(X , n_components=None , mean_sub=True, zscore=False, plot_pref=False , n_PCs_toPlot=2):

    if mean_sub and not zscore:
        X = X - np.mean(X, axis=0)
    if zscore:
        # X = scipy.stats.zscore(X, axis=0)
        X = X - np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = X / stds[None,:]
    else:
        stds = None
    
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
    
    return components , scores , decomp.explained_variance_ratio_ , stds


def torch_pca(  X, 
                device='cpu', 
                mean_sub=True, 
                zscore=False, 
                rank=None, 
                return_cpu=True, 
                return_numpy=False):
    """
    Principal Components Analysis for PyTorch.
    If using GPU, then call torch.cuda.empty_cache() after.
    RH 2021

    Args:
        X (torch.Tensor or np.ndarray):
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
    
    if isinstance(X, torch.Tensor) == False:
        X = torch.from_numpy(X).to(device)
    elif X.device != device:
            X = X.to(device)
            
    if mean_sub and not zscore:
        X = X - torch.mean(X, dim=0)
    if zscore:
        X = X - torch.mean(X, dim=0)
        stds = torch.std(X, dim=0)
        X = X / stds[None,:]        
        
    if rank is None:
        rank = X.shape[1]
    
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
        
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return components, scores, singVals, EVR


def torch_pca_batched(  X, 
                        n_batches=3,
                        batching_mode='random',
                        batch_idx=None,
                        device='cpu', 
                        mean_sub=True, 
                        zscore=False, 
                        rank=None, 
                        return_cpu=True, 
                        return_numpy=False):
    """
    Principal Components Analysis for PyTorch.
    If using GPU, then call torch.cuda.empty_cache() after.
    RH 2021

    Args:
        X (torch.Tensor or np.ndarray):
            Data to be decomposed.
            2-D array. Columns are features, rows are samples.
            PCA will be performed column-wise.
        n_batches (int):
            Number of batches to use.
        batching_mode (str):
            How to batch the data.
            'random' - Indices are randomly selected.
            'sequential' - Indices are sequentially selected.
        batch_idx (list of list of ints):
            Indices to use for each batch.
            If None, then batch_idx will be generated.
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
    
    if isinstance(X, torch.Tensor) == False:
        X = torch.from_numpy(X).to(device)
    elif X.device != device:
            X = X.to(device)
            
    if mean_sub and not zscore:
        X = X - torch.mean(X, dim=0)
    if zscore:
        X = X - torch.mean(X, dim=0)
        stds = torch.std(X, dim=0)
        X = X / stds[None,:]        
        
    if rank is None:
        rank = X.shape[1]

    if batch_idx is None:
        X_idx = np.arange(X.shape[0])
        l = X.shape[0]
        r = n_batches-l%n_batches # remainder
        test_rp = np.random.permutation(X_idx)
        test_pad = np.concatenate((test_rp,[np.nan]*r))
        batches = test_pad.reshape(n_batches,-1)
        batches_list = [[batch[~np.isnan(batch)]] for batch in batches]
    
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
        
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return components, scores, singVals, EVR