from pathlib import Path
import tkinter as tk
import PIL
from PIL import ImageTk
import csv
import warnings
import time

from matplotlib import pyplot as plt
import numpy as np
import cv2


###############
#### PLOTS ####
###############

def plot_image_grid(images, labels=None, grid_shape=(10,10), show_axis='off', cmap=None, kwargs_subplots={}, kwargs_imshow={}):
    """
    Plot a grid of images
    RH 2021

    Args:
        images (list of 2D arrays or a 3D array):
            List of images or a 3D array of images
             where the first dimension is the number of images
        labels (list of strings):
            List of labels to be displayed in the grid
        grid_shape (tuple):
            Shape of the grid
        show_axis (str):
            Whether to show axes or not
        cmap (str):
            Colormap to use
        kwargs_subplots (dict):
            Keyword arguments for subplots
        kwargs_imshow (dict):
            Keyword arguments for imshow
    
    Returns:
        fig:
            Figure
        axs:
            Axes
    """
    if cmap is None:
        cmap = 'viridis'

    fig, axs = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], **kwargs_subplots)
    axs_flat = axs.flatten(order='F') if isinstance(axs, np.ndarray) else [axs]
    for ii, ax in enumerate(axs_flat[:len(images)]):
        ax.imshow(images[ii], cmap=cmap, **kwargs_imshow);
        if labels is not None:
            ax.set_title(labels[ii]);
        ax.axis(show_axis);
    return fig, axs


def widget_toggle_image_stack(images, labels=None, clim=None, figsize=None):
    """
    Scrub through iamges in a stack using a slider.
    Requires %matplotlib notebook or %matplotlib widget
    RH 2022

    Args:
        images (list of 2D arrays):
            List of images
        clim (tuple):
            Limits of the colorbar
    """
    from ipywidgets import interact, widgets

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    imshow_FOV = ax.imshow(
        images[0],
#         vmax=clim[1]
    )

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        imshow_FOV.set_data(images[i_frame])
        imshow_FOV.set_clim(clim)
        if labels is not None:
            ax.set_title(labels[i_frame])


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def display_toggle_image_stack(images, image_size=None, clim=None, interpolation='nearest'):
    """
    Display images in a slider using Jupyter Notebook.
    RH 2023

    Args:
        images (list of numpy arrays or PyTorch tensors):
            List of images as numpy arrays or PyTorch tensors
        image_size (tuple of ints, optional):
            Tuple of (width, height) for resizing images.
            If None (default), images are not resized.
        clim (tuple of floats, optional):
            Tuple of (min, max) values for scaling pixel intensities.
            If None (default), min and max values are computed from the images
             and used as bounds for scaling.
        interpolation (string, optional):
            String specifying the interpolation method for resizing.
            Options: 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'.
            Uses the Image.Resampling.* methods from PIL.
    """
    from IPython.display import display, HTML
    import numpy as np
    import base64
    from PIL import Image
    from io import BytesIO
    import torch
    import datetime
    import hashlib
    import sys
    
    def normalize_image(image, clim=None):
        """Normalize the input image using the min-max scaling method. Optionally, use the given clim values for scaling."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if clim is None:
            clim = (np.min(image), np.max(image))

        norm_image = (image - clim[0]) / (clim[1] - clim[0])
        norm_image = np.clip(norm_image, 0, 1)
        return (norm_image * 255).astype(np.uint8)
    def resize_image(image, new_size, interpolation):
        """Resize the given image to the specified new size using the specified interpolation method."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(new_size, resample=interpolation)
        return np.array(resized_image)
    def numpy_to_base64(numpy_array):
        """Convert a numpy array to a base64 encoded string."""
        img = Image.fromarray(numpy_array.astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("ascii")
    def process_image(image):
        """Normalize, resize, and convert image to base64."""
        # Normalize image
        norm_image = normalize_image(image, clim)

        # Resize image if requested
        if image_size is not None:
            norm_image = resize_image(norm_image, image_size, interpolation_method)

        # Convert image to base64
        return numpy_to_base64(norm_image)


    # Check if being called from a Jupyter notebook
    if 'ipykernel' not in sys.modules:
        raise RuntimeError("This function must be called from a Jupyter notebook.")

    # Create a dictionary to map interpolation string inputs to Image objects
    interpolation_methods = {
        'nearest': Image.Resampling.NEAREST,
        'box': Image.Resampling.BOX,
        'bilinear': Image.Resampling.BILINEAR,
        'hamming': Image.Resampling.HAMMING,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
    }

    # Check if provided interpolation method is valid
    if interpolation not in interpolation_methods:
        raise ValueError("Invalid interpolation method. Choose from 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', or 'lanczos'.")

    # Get the actual Image object for the specified interpolation method
    interpolation_method = interpolation_methods[interpolation]

    # Generate a unique identifier for the slider
    slider_id = hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest()

    # Process all images in the input list
    base64_images = [process_image(img) for img in images]

    # Get the image size for display
    image_size = images[0].shape[:2] if image_size is None else image_size

    # Generate the HTML code for the slider
    html_code = f"""
    <div>
        <input type="range" id="imageSlider_{slider_id}" min="0" max="{len(base64_images) - 1}" value="0">
        <img id="displayedImage_{slider_id}" src="data:image/png;base64,{base64_images[0]}" style="width: {image_size[1]}px; height: {image_size[0]}px;">
        <span id="imageNumber_{slider_id}">Image 0/{len(base64_images) - 1}</span>
    </div>

    <script>
        (function() {{
            let base64_images = {base64_images};
            let current_image = 0;
    
            function updateImage() {{
                let slider = document.getElementById("imageSlider_{slider_id}");
                current_image = parseInt(slider.value);
                let displayedImage = document.getElementById("displayedImage_{slider_id}");
                displayedImage.src = "data:image/png;base64," + base64_images[current_image];
                let imageNumber = document.getElementById("imageNumber_{slider_id}");
                imageNumber.innerHTML = "Image " + current_image + "/{len(base64_images) - 1}";
            }}
            
            document.getElementById("imageSlider_{slider_id}").addEventListener("input", updateImage);
        }})();
    </script>
    """

    display(HTML(html_code))


def plot_to_image(fig, keep_alpha=True):
    """
    Convert a matplotlib figure to a numpy array image.
    Recommendations:
        - Use fig.tight_layout() to avoid overlapping subplots
        - Use ax.margins(0.0) to remove margins
        - Use ax.axis('off') to hide axes
    Output will be RGBA format, shape (width, height, 4).
    RH 2022

    Args:
        fig (matplotlib.figure):
            figure to convert to numpy array.
    
    Returns:
        image (np.array):
            numpy array image.
            shape: (height, width, n_channels:4)
    """

    fig.canvas.draw()
    if keep_alpha:
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[...,(1,2,3,0)]
    else:
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image


def shaded_error_bar(
    data,
    x=None,
    ax=None,
    reduction_center=np.nanmean,
    reduction_fill=np.nanstd,
    alpha_fill=0.3,
    color_center=None,
    color_fill=None,
    kwargs_center=None,
    kwargs_fill=None,
):
    """
    Plot a shaded error bar.
    RH 2022

    Args:
        data (np.array):
            Data to plot.
            Should be 2D array with shape (n_samples, n_features).
        x (np.array):
            x-axis values.
            Optional.
        ax (matplotlib.axes):
            Axes to plot on.
            Optional.
        reduction_center (function):
            Function to reduce data along axis=1 to get the 
             center line.
            Must have accept 'axis=1' as an argument.
        reduction_fill (function):
            Function to reduce data along axis=1 to get the
             fill.
            Must have accept 'axis=1' as an argument.
        alpha_fill (float):
            Alpha value for the fill.
        color_center (str):
            Color for the center line.
        color_fill (str):
            Color for the fill.
        kwargs_center (dict):
            Keyword arguments for the center line.
            Will be passed to ax.plot().
        kwargs_fill (dict):
            Keyword arguments for the fill.
            Will be passed to ax.fill_between().

    Returns:
        ax (matplotlib.axes):
            Axes.
    """
    if ax is None:
        ax = plt.gca()
    
    if x is None:
        x = np.arange(data.shape[0])
    
    if color_center is None:
        color_center = ax._get_lines.get_next_color()
    
    if color_fill is None:
        color_fill = color_center

    y_center = reduction_center(data, axis=1)
    y_error = reduction_fill(data, axis=1)  

    kwargs_center = {} if kwargs_center is None else kwargs_center
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill

    ax.plot(x, y_center, color=color_center, **kwargs_center)
    ax.fill_between(
        x,
        y_center - y_error,
        y_center + y_error,
        color=color_fill,
        alpha=alpha_fill,
        **kwargs_fill,
    )
    return ax


###############
### Helpers ###
###############

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


def rand_cmap(
    nlabels, 
    type='bright', 
    first_color_black=False, 
    last_color_black=False,
    verbose=True,
    under=[0,0,0],
    over=[0.5,0.5,0.5],
    bad=[0.9,0.9,0.9],
    ):
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

    assert nlabels > 0, 'Number of labels must be greater than 0'


    if type not in ('bright', 'soft', 'random'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

#     # Generate color map for bright colors, based on hsv
#     if type == 'bright':
#         randHSVcolors = [(np.random.uniform(low=0.0, high=1),
#                           np.random.uniform(low=0.2, high=1),
# #                           np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]
#                           np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

#         # Convert HSV list to RGB
#         randRGBcolors = []
#         for HSVcolor in randHSVcolors:
#             randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

    randRGBcolors = np.random.rand(nlabels, 3)
    if type == 'bright':
        randRGBcolors = randRGBcolors / np.max(randRGBcolors, axis=1, keepdims=True)

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

    random_colormap.set_bad(bad)
    random_colormap.set_over(over)
    random_colormap.set_under(under)

    return random_colormap


def simple_cmap(
    colors=[
        [1,0,0],
        [1,0.6,0],
        [0.9,0.9,0],
        [0.6,1,0],
        [0,1,0],
        [0,1,0.6],
        [0,0.8,0.8],
        [0,0.6,1],
        [0,0,1],
        [0.6,0,1],
        [0.8,0,0.8],
        [1,0,0.6],
    ],
    under=[0,0,0],
    over=[0.5,0.5,0.5],
    bad=[0.9,0.9,0.9],
    input_values=[0,1],
    N=256,
    name='none'
):
    """Create a colormap from a sequence of rgb values.
    Stolen with love from Alex (https://gist.github.com/ahwillia/3e022cdd1fe82627cbf1f2e9e2ad80a7ex)
    
    Args:
        colors (list):
            List of RGB values
        name (str):
            Name of the colormap

    Returns:
        cmap:
            Colormap

    Demo:
    cmap = simple_cmap([(1,1,1), (1,0,0)]) # white to red colormap
    cmap = simple_cmap(['w', 'r'])         # white to red colormap
    cmap = simple_cmap(['r', 'b', 'r'])    # red to blue to red
    """
    from matplotlib.colors import LinearSegmentedColormap, colorConverter

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # convert colors to rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    ## prep input_values
    input_values = np.array(input_values)
    ### assert monotonic
    if not np.all(np.diff(input_values) > 0):
        raise ValueError('input_values must be monotonically increasing')
    fn_norm = lambda x: (x - input_values.min()) / np.ptp(input_values)
    input_values_norm = fn_norm(input_values)
    ### assert either n_inputs is 2 or n_colors
    if len(input_values) == 2:
        input_values_norm = np.linspace(0., 1., n_colors, endpoint=True)
    elif len(input_values) != n_colors:
        raise ValueError('length of input_values must be either 2 or equal to the length of colors')

    # set up colormap
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, (r, g, b) in enumerate(colors):
        idx = input_values_norm[i]
        # print(idx)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    cmap = LinearSegmentedColormap(
        name=name, 
        segmentdata={k: tuple(v) for k, v in cdict.items()},
        N=N,
    )
                                   
    cmap.set_bad(bad)
    cmap.set_over(over)
    cmap.set_under(under)

    import matplotlib
    class LSC_norm(matplotlib.colors.LinearSegmentedColormap):
        def __init__(self, name=name, N=N):
            super().__init__(name=name, segmentdata={k: tuple(v) for k, v in cdict.items()}, N=N)
            self.set_bad(bad)
            self.set_over(over)
            self.set_under(under)

        def __call__(self, X, alpha=None, bytes=False):
            X_norm = fn_norm(X)
            return super().__call__(X_norm, alpha=alpha, bytes=bytes)
        
    cmap_out = LSC_norm(name=name, N=N)

    return cmap_out

class Cmap_conjunctive:
    """
    Combines multiple colormaps into a single colormap by
     multiplying their values together.
    RH 2022
    """
    def __init__(
        self, 
        cmaps, 
        dtype_out=int, 
        normalize=False,
        normalization_range=[0,255],
        name='cmap_conjunctive',
    ):
        """
        Initialize the colormap transformer.

        Args:
            cmaps (list):
                List of colormaps to combine.
                Should be a list of matplotlib.colors.LinearSegmentedColormap objects.
            dtype (np.dtype):
                Data type of the output colormap.
            normalize (bool):
                Whether to normalize the inputs to (0,1) for the input to cmaps.
            normalization_range (list):
                Range to normalize the outputs to.
                Should be a list of two numbers.
            name (str):
                Name of the colormap.
        """
        import matplotlib

        ## Check inputs
        assert isinstance(cmaps, list), 'cmaps must be a list.'
        assert all([isinstance(cmap, matplotlib.colors.LinearSegmentedColormap) for cmap in cmaps]), 'All elements of cmaps must be matplotlib.colors.LinearSegmentedColormap objects.'

        self.cmaps = cmaps
        self.dtype_out = dtype_out
        self.name = name
        self.normalize = normalize
        self.normalization_range = normalization_range

        self.n_cmaps = len(self.cmaps)

        self.fn_conj_cmap = lambda x: np.prod(np.stack([cmap(x_i) for cmap,x_i in zip(self.cmaps, x.T)], axis=0), axis=0)

    def __call__(self, x):
        """
        Apply the colormap to the input data.

        Args:
            x (np.ndarray):
                Input data.
                Should be a numpy array of shape (n_samples, n_cmaps).
                If normalize==True, then normalization is applied to
                 each column of x separately.

        Returns:
            (np.ndarray):
                Colormapped data.
                Will be a numpy array of shape (n_samples, 4).
        """
        assert isinstance(x, np.ndarray), 'x must be a numpy array of shape (n_samples, n_cmaps).'

        ## Make array 2D
        if x.ndim == 1:
            x = x[None,:]
        assert x.shape[1] == self.n_cmaps, 'x.shape[1] must match the number of cmaps.'

        ## Normalize x
        if self.normalize:
            assert x.shape[1] > 1, 'x must have more than one row to normalize.'
            x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True))

        ## Get colors
        colors = self.fn_conj_cmap(x)
        colors = (colors * (self.normalization_range[1] - self.normalization_range[0]) + self.normalization_range[0]).astype(self.dtype_out)

        return colors


class Figure_Saver:
    """
    Class for saving figures
    RH 2022
    """
    def __init__(
        self,
        dir_save: str=None,
        format_save: list=['png', 'svg'],
        kwargs_savefig: dict={
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': True,
            'dpi': 300,
        },
        overwrite: bool=False,
        mkdir: bool=True,
        verbose: int=1,
    ):
        """
        Initializes Figure_Saver object

        Args:
            dir_save (str):
                Directory to save the figure. Used if path_config is None.
                Must be specified if path_config is None.
            format (list of str):
                Format(s) to save the figure. Default is 'png'.
                Others: ['png', 'svg', 'eps', 'pdf']
            overwrite (bool):
                If True, then overwrite the file if it exists.
            kwargs_savefig (dict):
                Keyword arguments to pass to fig.savefig().
            verbose (int):
                Verbosity level.
                0: No output.
                1: Warning.
                2: All info.
        """
        self.dir_save = str(Path(dir_save).resolve().absolute()) if dir_save is not None else None

        assert isinstance(format_save, list), "RH ERROR: format_save must be a list of strings"
        assert all([isinstance(f, str) for f in format_save]), "RH ERROR: format_save must be a list of strings"
        self.format_save = format_save

        assert isinstance(kwargs_savefig, dict), "RH ERROR: kwargs_savefig must be a dictionary"
        self.kwargs_savefig = kwargs_savefig

        self.overwrite = overwrite
        self.mkdir = mkdir
        self.verbose = verbose

    def save(
        self,
        fig,
        path_save: str=None,
        dir_save: str=None,
        name_file: str=None,
    ):
        """
        Save the figures.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            path_save (str):
                Path to save the figure.
                Should not contain suffix.
                If None, then the dir_save must be specified here or in
                 the initialization and name_file must be specified.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure), "RH ERROR: fig must be a matplotlib.figure.Figure"

        ## Get path_save
        if path_save is not None:
            assert len(Path(path_save).suffix) == 0, "RH ERROR: path_save must not contain suffix"
            path_save = [str(Path(path_save).resolve()) + '.' + f for f in self.format_save]
        else:
            assert (dir_save is not None) or (self.dir_save is not None), "RH ERROR: dir_save must be specified if path_save is None"
            assert name_file is not None, "RH ERROR: name_file must be specified if path_save is None"

            ## Get dir_save
            dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

            ## Get figure title
            if name_file is None:
                titles = [a.get_title() for a in fig.get_axes() if a.get_title() != '']
                name_file = '.'.join(titles)
            path_save = [str(Path(dir_save) / (name_file + '.' + f)) for f in self.format_save]

        ## Make directory
        if self.mkdir:
            Path(path_save[0]).parent.mkdir(parents=True, exist_ok=True)

        ## Save figure
        for path, form in zip(path_save, self.format_save):
            if Path(path).exists():
                if self.overwrite:
                    print(f'RH Warning: Overwriting file. File: {path} already exists.') if self.verbose > 0 else None
                else:
                    print(f'RH Warning: Not saving anything. File exists and overwrite==False. {path} already exists.') if self.verbose > 0 else None
                    return None
            print(f'FR: Saving figure {path} as format(s): {form}') if self.verbose > 1 else None
            fig.savefig(path, format=form, **self.kwargs_savefig)

    def save_batch(
        self,
        figs,
        dir_save: str=None,
        names_files: str=None,
    ):
        """
        Save all figures in a list.

        Args:
            figs (list of matplotlib.figure.Figure):
                Figures to save.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(figs, list), "RH ERROR: figs must be a list of matplotlib.figure.Figure"
        assert all([isinstance(fig, plt.Figure) for fig in figs]), "RH ERROR: figs must be a list of matplotlib.figure.Figure"

        ## Get dir_save
        dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

        for fig, name_file in zip(figs, names_files):
            self.save(fig, name_file=name_file, dir_save=dir_save)

    def __call__(
        self,
        fig,
        name_file: str=None,
        path_save: str=None,
        dir_save: str=None,
    ):
        """
        Calls save() method.
        """
        self.save(fig, path_save=path_save, name_file=name_file, dir_save=dir_save)

    def __repr__(self):
        return f"Figure_Saver(dir_save={self.dir_save}, format={self.format_save}, overwrite={self.overwrite}, kwargs_savefig={self.kwargs_savefig}, verbose={self.verbose})"


#############################################
########### Interactive plots ###############
#############################################


class Select_ROI:
    """
    A simple GUI to select ROIs from an image.
    Uses matplotlib and ipywidgets.
    Only works in a Jupyter notebook.
    Select regions of interest in an image using matplotlib.
    Use %matplotlib notebook or qt backend to use this.
    It currently uses cv2.polylines to draw the ROIs.
    Output is self.mask_frames
    RH 2021
    """

    def __init__(self, image, kwargs_subplots={}, kwargs_imshow={}, backend='module://ipympl.backend_nbagg'):
        """
        Initialize the class

        Args:
            im:
                Image to select the ROI from
        """
        from ipywidgets import widgets
        import IPython.display as Disp
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        ## set jupyter notebook to use interactive matplotlib.
        ## Available backends: ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
        ## Set backend. Equivalent to %matplotlib widget
        # mpl.use('module://ipympl.backend_nbagg')
        mpl.use(backend)
        plt.ion()

        ## Set variables                
        self._img_input = image.copy()
        self.selected_points = []
        self._selected_points_last_ROI = []
        self._completed_status = False

        ## Prepare figure
        self._fig, self._ax = plt.subplots(**kwargs_subplots)
        self._img_current = self._ax.imshow(self._img_input.copy(), **kwargs_imshow)

        ## Connect the click event
        self._buttonRelease = self._fig.canvas.mpl_connect('button_release_event', self._onclick)
        ## Make and connect the buttons
        disconnect_button = widgets.Button(description="Confirm ROI")
        new_ROI_button = widgets.Button(description="New ROI")
        Disp.display(disconnect_button)
        Disp.display(new_ROI_button)
        disconnect_button.on_click(self._disconnect_mpl)
        new_ROI_button.on_click(self._new_ROI)

    def _poly_img(self, img, pts):
        """
        Draw a polygon on an image.
        """
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            img=img, 
            pts=[pts],
            isClosed=True,
            color=(255,255,255,),
            thickness=2
        )
        return img

    def _onclick(self, event):
        """
        When the mouse is clicked, add the point to the list.
        """
        ## If the click is outside the image, ignore it
        if (event.xdata is None) or (event.ydata is None):
            return None
        self._selected_points_last_ROI.append([event.xdata, event.ydata])
        if len(self._selected_points_last_ROI) > 1:
            self._fig
            im = self._img_input.copy()
            for ii in range(len(self.selected_points)):
                im = self._poly_img(im, self.selected_points[ii])
            im = self._poly_img(im, self._selected_points_last_ROI)
            self._img_current.set_data(im)
        self._fig.canvas.draw()


    def _disconnect_mpl(self, _):
        """
        Disconnect the click event and collect the points.
        """
        import skimage.draw

        self.selected_points.append(self._selected_points_last_ROI)

        self._fig.canvas.mpl_disconnect(self._buttonRelease)
        self._completed_status = True
        
        self.mask_frames = []
        for ii, pts in enumerate(self.selected_points):
            pts = np.array(pts)
            mask_frame = np.zeros((self._img_input.shape[0], self._img_input.shape[1]))
            pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
            mask_frame[pts_y, pts_x] = 1
            mask_frame = mask_frame.astype(np.bool_)
            self.mask_frames.append(mask_frame)
        print(f'mask_frames computed')

    def _new_ROI(self, _):
        """
        Start a new ROI.
        """
        self.selected_points.append(self._selected_points_last_ROI)
        self._selected_points_last_ROI = []
        
    
class Image_labeler:
    """
    A simple graphical interface for labeling image classes.
    Use this class with a context manager to ensure that the
     window is closed properly. See demo below.
    This class provides a tkinter window which displays images
     from a provided numpy array one by one and lets you classify
     each image by pressing a key. 
    The title of the window is the image index.
    The classification label and image index are stored as
     self.labels_ and saved to a CSV file in self.path_csv.
    RH 2023

    Demo:
    with Image_labeler(images, start_index=0, resize_factor=4.0, key_end='Escape') as labeler:
        labeler.run()
    path_csv, labels = labeler.path_csv, labeler.labels_

    Args:
        images (np.ndarray): 
            A numpy array of images.
            Either 3D: (n_images, height, width) or
             4D: (n_images, height, width, n_channels).
            Images should be scaled between 0 and 255 and will be
             converted to uint8.
        start_index (int): 
            The index of the first image to display. Default is 0.
        path_csv (str): 
            A string of the path to the CSV file for saving results.
            If None, results will not be saved.
        save_csv (bool):
            A boolean indicating whether to save the results to a CSV.
        resize_factor (float): 
            A scaling factor indicating the fractional change in 
             image size. Default is 1.0 (no change).
        key_end (str): 
            A string of the key to press to end the session.
        key_prev (str):
            A string of the key to press to go back to the previous
             image.
        key_next (str):
            A string of the key to press to go to the next image.
        normalize_images (bool):
            A boolean indicating whether to normalize the images
             between min and max values. Default is True.
        verbose (bool):
            A boolean indicating whether to print status updates.
    """

    def __init__(
        self, 
        image_array: np.ndarray, 
        start_index: int=0,
        path_csv: str=None, 
        save_csv: bool=True,
        resize_factor: float=10.0, 
        normalize_images: bool=True,
        verbose: bool=True,
        key_end: str='Escape', 
        key_prev: str='Left',
        key_next: str='Right',
    ):
        """
        Initializes class with images, path to save csv and UI
         elements.
        Binds keys for classifying images and ending the session.
        """
        import tempfile
        import datetime
        ## Set attributes
        self.images = image_array
        self._resize_factor = resize_factor
        self._index = start_index - 1  ## -1 because we increment before displaying
        self.path_csv = path_csv if path_csv is not None else str(Path(tempfile.gettempdir()) / ('labels_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'))
        self._save_csv = save_csv
        self.labels_ = {}
        self._img_tk = None
        self._key_end = key_end if key_end is not None else None
        self._key_prev = key_prev if key_prev is not None else None
        self._key_next = key_next if key_next is not None else None
        self._normalize_images = normalize_images
        self._verbose = verbose

        self.__call__ = self.run
        
    def run(self):
        """
        Runs the image labeler. Opens a tkinter window and displays
         the first image.
        """
        try:
            self._root = tk.Tk()
            self._img_label = tk.Label(self._root)
            self._img_label.pack()

            ## Bind keys
            self._root.bind("<Key>", self.classify)
            self._root.bind('<Key-' + self._key_end + '>', self.end_session) if self._key_end is not None else None
            self._root.bind('<Key-' + self._key_prev + '>', self.prev_img) if self._key_prev is not None else None
            self._root.bind('<Key-' + self._key_next + '>', self.next_img) if self._key_next is not None else None

            ## Start the session
            self.next_img()
            self._root.mainloop()
        except Exception as e:
            warnings.warn('Error initializing image labeler: ' + str(e))

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_session(None)

    def next_img(self, event=None):
        """Displays the next image in the array, and resizes the image."""
        ## Display the image
        ### End the session if there are no more images
        self._index += 1
        if self._index < len(self.images):
            im = self.images[self._index]
            im = (im / np.max(im)) * 255 if self._normalize_images else im
            pil_img = PIL.Image.fromarray(np.uint8(im))  ## Convert to uint8 and PIL image
            ## Resize image
            width, height = pil_img.size
            new_width = int(width * self._resize_factor)
            new_height = int(height * self._resize_factor)
            pil_img = pil_img.resize((new_width, new_height), resample=PIL.Image.LANCZOS)
            ## Display image
            self._img_tk = ImageTk.PhotoImage(pil_img)
            self._img_label.image = self._img_tk  # keep a reference to the PhotoImage object
            self._img_label.config(image=self._img_label.image)
        else:
            self.end_session(None)
        
        self._root.title(str(self._index))  # update the window title to the current image index

    def prev_img(self, event=None):
        """
        Displays the previous image in the array.
        """
        self._index -= 2
        self.next_img()

    def classify(self, event):
        """
        Adds the current image index and pressed key as a label.
        Then saves the results and moves to the next image.

        Args:
            event (tkinter.Event):
                A tkinter event object.
        """
        label = event.char
        if label != '':
            print(f'Image {self._index}: {label}') if self._verbose else None
            self.labels_.update({self._index: str(label)})  ## Store the label
            self.save_classification() if self._save_csv else None ## Save the results
            self.next_img()  ## Move to the next image

    def end_session(self, event):
        """Ends the classification session by destroying the tkinter window."""
        self._img_tk = None
        self._root.destroy() if self._root is not None else None
        self._root = None
        
        import gc
        gc.collect()
        gc.collect()

    def save_classification(self):
        """
        Saves the classification results to a CSV file.
        This function does not append, it overwrites the entire file.
        The file contains two columns: 'image_index' and 'label'.
        """
        ## make directory if it doesn't exist
        Path(self.path_csv).parent.mkdir(parents=True, exist_ok=True)
        ## Save the results
        with open(self.path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('image_index', 'label'))
            writer.writerows(self.labels_.items())

    def get_labels(self, kind='dict',):
        """
        Returns a dictionary of the labels.
        The keys are the image indices and the values are the labels.

        Args:
            kind (str):
                A string indicating the type of object to return.
                'dict': {idx: label, idx: label, ...}
                'list': [(idx, label), (idx, label), ...]
                'dataframe': {'index': [idx, idx, ...], 'label': [label, label, ...]}
                    This can be converted to a pandas dataframe with:
                     pd.DataFrame(self.get_labels('dataframe'))
        """
        ## if the dict is empty, return None
        if len(self.labels_) == 0:
            return None
        
        if kind == 'dict':
            # out = dict(self.labels_)
            # ## Check for duplicate indices
            # if len(out) != len(self.labels_):
            #     warnings.warn('Duplicate indices found in labels. Only the last label for each index is returned.')
            # return out
            return self.labels_
        elif kind == 'list':
            # return self.labels_
            return self.labels_.items()
        elif kind == 'dataframe':
            # return {'index': np.array([x[0] for x in self.labels_], dtype=np.int64), 'label': np.array([x[1] for x in self.labels_])}
            return {'index': np.array(list(self.labels_.keys()), dtype=np.int64), 'label': np.array(list(self.labels_.values()), dtype=str)}


def rgb_to_hex(r, g, b):
    if isinstance(r, float):
        r, g, b = (int(v*255) for v in (r,g,b))
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


########################################################################################################################
########################################## OTHER PLOTTING LIBRARIES ####################################################
########################################################################################################################

def export_svg_hv_bokeh(obj, path_save):
    """
    Save a scatterplot from holoviews as an svg file. Call
    bokeh.io.output_notebook() if calling from a jupyter notebook.
    RH 2023

    Args:
        obj (holoviews.plotting.bokeh.ElementPlot):
            Holoviews plot object.
        path_save (str):
            Path to save the svg file.
    """
    import holoviews as hv
    import bokeh
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    bokeh.io.export_svgs(plot_state, filename=path_save)

def plot_lines_bokeh(x=None, y=None, figsize=(500,500)):
    import bokeh.plotting as bp
    # y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    # x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    if (x is not None) and (y is None):
        y = x
        y = y[:,None] if y.ndim == 1 else y
        x = np.arange(y.shape[0])
    elif (x is None) and (y is not None):
        x = np.arange(y.shape[0])
        y = y[:,None] if y.ndim == 1 else y
    elif (x is None) and (y is None):
        raise ValueError('x and y cannot both be None')
        
    cmap = simple_cmap(
        colors=[[0,0,1],[0,1,0],[1,0,0]],
    )
    p = bp.figure()
    for i_line in range(y.shape[1]):
        c = rgb_to_hex(*cmap(i_line / (y.shape[1]-1))[:3]) if y.shape[1] > 1 else 'black'
        p.line(
            x=x, 
            y=y[:,i_line], 
            color=c,
            line_width=2,
        )
        
    p.frame_height=figsize[0]
    p.frame_width=figsize[1]
    bp.show(p)
    return p