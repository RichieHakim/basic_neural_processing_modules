import sklearn.decomposition
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def simple_pca(X , n_components=None , plot_pref=False , n_PCs_toPlot=2):
    
    if n_components is None:
        n_components = X.shape[1]
    decomp = sklearn.decomposition.PCA(n_components=n_components)
    loadings = decomp.fit_transform(X)

    scores = X.T @ loadings
    
    if plot_pref:
        fig , axs = plt.subplots(3 , figsize=(7,12))
        axs[0].plot(np.arange(n_components)+1,
                    decomp.explained_variance_ratio_)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('component #')
        axs[0].set_ylabel('explained variance ratio')

        axs[1].plot(loadings[:,:n_PCs_toPlot])
        axs[1].set_xlabel('sample num')
        axs[1].set_ylabel('a.u.')

        axs[2].plot(scores[:,:n_PCs_toPlot])
        axs[2].set_xlabel('feature num')
        axs[2].set_ylabel('score')
    
    return loadings , scores



def make_xcorrMat(vector_set1 , vector_set2=None):
    if vector_set2 is None:
        vector_set2 = vector_set1
    return (scipy.stats.zscore(vector_set1, axis=0).T @ scipy.stats.zscore(vector_set2, axis=0)) / ((vector_set1.shape[0] + vector_set2.shape[0])/2)
