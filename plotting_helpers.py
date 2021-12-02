from matplotlib import pyplot as plt
import numpy as np

def get_subplot_indices(axs):
    """
    Returns the subscript indices of the axes of a subplots figure.
    Basically perfoms an ind2sub operation on the axis number.
    RH 2021

    Args:
        axs:
            list of axes
    
    Returns:
        list of indices
    
    """
    out_array = np.stack(np.unravel_index(np.arange(np.prod(axs.shape)), axs.shape, order='F'), axis=-1)
    return [tuple(ii) for ii in out_array]

def plot_image_grid(images, grid_shape=(10,10)):
    """
    Plot a grid of images
    RH 2021

    Args:
        images (list of 2D arrays or a 3D array):
            List of images or a 3D array of images
             where the first dimension is the number of images
        grid_shape (tuple):
            Shape of the grid
    
    Returns:
        fig:
            Figure
        axs:
            Axes
    """
    fig, axs = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1])
    idx_axs = get_subplot_indices(axs)
    for ii,idx_ax in enumerate(idx_axs):
        axs[idx_ax].imshow(images[ii])
        axs[idx_ax].axis('off')
    return fig, axs

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
#                           np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
#                           np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        fig, ax = plt.subplots(1, 1, figsize=(6, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

