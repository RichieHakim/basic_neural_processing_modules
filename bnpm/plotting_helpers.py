from matplotlib import pyplot as plt
import numpy as np

import cv2

from tqdm.notebook import tqdm

from ipywidgets import widgets
import IPython.display as Disp

from pathlib import Path


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
    idx_axs = get_subplot_indices(axs)
    for ii,idx_ax in enumerate(idx_axs):
        axs[idx_ax].imshow(images[ii], cmap=cmap, **kwargs_imshow);
        if labels is not None:
            axs[idx_ax].set_title(labels[ii]);
        axs[idx_ax].axis(show_axis);
    return fig, axs


def display_toggle_image_stack(images, labels=None, clim=None, figsize=None):
    """
    Scrub through iamges in a stack using a slider.
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
    name='none'):
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

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    cmap = LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})
                                   
    cmap.set_bad(bad)
    cmap.set_over(over)
    cmap.set_under(under)

    return cmap

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

def savefig(path, dpi=300, mkdir=True, **kwargs_savefig):
    """
    Save figure.
    RH 2022

    Args:
        path (str):
            Path to save figure to.
            Format is inferred from path suffix.
        dpi (int):
            DPI of figure.
        **kwargs_savefig:
            Keyword arguments to pass to `matplotlib.pyplot.savefig`.
    """

    assert len(Path(path).suffix) > 0, 'RH Error: Path must have a suffix'
    kind = Path(path).suffix[1:]

    if mkdir:
        Path(path).parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(path, format=kind, dpi=dpi, **kwargs_savefig)


class Figure_Saver:
    """
    Class for saving figures
    RH 2022
    """
    def __init__(
        self,
        dir_save: str=None,
        format_save: list=['png'],
        kwargs_savefig: dict={
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': True,
            'dpi': 300,
        },
        overwrite: bool=False,
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
        path_save: str=None,
        name_file: str=None,
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

    def __init__(self, image, kwargs_subplots={}, kwargs_imshow={}):
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
        ## equivalent to %matplotlib notebook
        mpl.use("nbagg")        
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
            mask_frame = mask_frame.astype(np.bool)
            self.mask_frames.append(mask_frame)
        print(f'mask_frames computed')

    def _new_ROI(self, _):
        """
        Start a new ROI.
        """
        self.selected_points.append(self._selected_points_last_ROI)
        self._selected_points_last_ROI = []
        
        
