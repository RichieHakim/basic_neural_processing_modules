from matplotlib import pyplot as plt
import numpy as np

import cv2

from ipywidgets import widgets
import IPython.display as Disp
import skimage.draw
import matplotlib

from matplotlib.colors import LinearSegmentedColormap, colorConverter


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


def simple_cmap(colors, name='none'):
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

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})

#############################################
################ Video ######################
#############################################

def play_video_cv2(array, frameRate, save_path=None, show=True, fourcc_code='MJPG', text=None, kwargs_text={}):
    """
    Play a video using OpenCV
    RH 2021

    Args:
        array:
            Either 3D array of images (frames x height x width)
             or a 4D array of images (frames x height x width x channels)
            Scaling assumed to be between 0 and 255
        frameRate:  
            Frame rate of the video (in Hz)
        save_path:
            Path to save the video
        show:   
            Whether to show the video or not
        fourcc_code:
            FourCC code for the codec
        text:
            Text to write on the video.
            If list, each element is on a different frame
        kwargs_text:
            Keyword arguments for text
    """
    wait_frames = max(int((1/frameRate)*1000), 1)
    if save_path is not None:
        size = tuple((np.flip(array.shape[1:])))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        print(f'saving to file {save_path}')
        writer = cv2.VideoWriter(save_path, fourcc, frameRate, size)

    if kwargs_text is None:
        kwargs_text = { 'org': (5, 15), 
                        'fontFace': 1, 
                        'fontScale': 1,
                        'color': (255, 255, 255), 
                        'thickness': 1}
    
    if array.dtype != 'uint8':
        array = array.astype('uint8')
         
    for i_frame, frame in enumerate(array):
        if array.shape[3] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:  
            frame = cv2.merge([frame, frame, frame])

        if text is not None:
            if isinstance(text, list):
                text_frame = text[i_frame]
            else:
                text_frame = text

            # frame = cv2.putText(frame, text, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, text_frame, **kwargs_text)
            
        if show:
            cv2.imshow('handle', np.uint8(frame))
            cv2.waitKey(wait_frames)
        if save_path is not None:
            writer.write(np.uint8(frame))
    if save_path is not None:
        writer.release()
        print('Video saved')
    if show:
        cv2.destroyWindow('handle')


#############################################
########### Interactive plots ###############
#############################################


class select_ROI:
    """
    Select a region of interest in an image using matplotlib.
    Use %matplotlib notebook or qt backend to use this.
    It currently uses cv2.polylines to draw the ROI.
    Output is self.mask_frame
    RH 2021
    """

    def __init__(self, im):
        """
        Initialize the class

        Args:
            im:
                Image to select the ROI from
        """
        self.im = im
        self.selected_points = []
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.im.copy())
        self.completed_status = False
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Confirm ROI")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, 
                      [pts],
                      True,
                      (255,255,255),
                      2)
        return img

    def onclick(self, event):
        self.selected_points.append([event.xdata, event.ydata])
        if len(self.selected_points) > 1:
            self.fig
            self.img.set_data(self.poly_img(self.im.copy(), self.selected_points))

    def disconnect_mpl(self, _):
        self.fig.canvas.mpl_disconnect(self.ka)
        self.completed_status = True
        
        pts = np.array(self.selected_points)
        self.mask_frame = np.zeros((self.im.shape[0], self.im.shape[1]))
        pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
        self.mask_frame[pts_y, pts_x] = 1
        self.mask_frame = self.mask_frame.astype(np.bool)
        print(f'mask_frame computed')
        