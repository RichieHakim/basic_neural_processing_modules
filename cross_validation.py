import matplotlib.pyplot as plt
import numpy as np

def make_cv_indices(cv, groups, cmap_cv=plt.cm.binary, cmap_data=plt.cm.binary, lw=20, plot_pref=True):

    """Create a sample plot for indices of a cross-validation object."""

    X = np.arange(len(groups))
    y = np.arange(len(groups))
    cv_idx = list(cv.split(X=X , y=y , groups=groups))
    
    n_splits = cv.n_splits
    # Generate the training/testing visualizations for each CV split
    if plot_pref:
        fig, ax = plt.subplots()

        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices, marker='_', lw=lw, cmap=cmap_cv,
                       vmin=-.2, vmax=1.2)

        ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                   c=groups, marker='_', lw=lw, cmap=cmap_data)

        # Formatting
        yticklabels = list(range(n_splits)) + ['group']
        ax.set(yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
               xlabel='Sample index', ylabel="CV iteration",
               ylim=[n_splits+1.2, -.2], xlim=[0, 108000])
        ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return cv_idx