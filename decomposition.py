import sklearn.decomposition
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def simple_pca(X , n_components=None , plot_pref=False , n_PCs_toPlot=2):
    
    if n_components is None:
        n_components = X.shape[1]
    decomp = sklearn.decomposition.PCA(n_components=n_components)
    decomp.fit_transform(X)
    components = decomp.components_
    scores = decomp.transform(X)

    # scores = X.T @ loadings # this shouldn't have to be done
    
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