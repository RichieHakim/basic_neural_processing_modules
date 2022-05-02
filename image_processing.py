import numpy as np
import cv2
import scipy as sp 


###############################################################################
## This block of code is used to initialize cv2.imshow
## This is necessary because importing av and decord 
##  will cause cv2.imshow to fail unless it is initialized.
## Obviously, this should be commented out when running on
##  systems that do not support cv2.imshow like servers.
## Also be sure to import BNPM before importing most other
##  modules.
test = np.zeros((1,300,400,3))
for frame in test:
    cv2.putText(frame, "Prepping CV2", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "to run after importing av and decord", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow('startup', frame)
    cv2.waitKey(100)
cv2.destroyWindow('startup')
###############################################################################


import av
import decord
# import cv2 

import copy
import torch
import torchvision
from tqdm.notebook import tqdm

from . import indexing


def find_registration_transformation(
    im_template, 
    im_moving,
    warp_mode='MOTION_HOMOGRAPHY',
    n_iter=5000,
    termination_eps=1e-10,
    mask=None,
    gaussFiltSize=1
):
    """
    Find the transformation between two images.
    Only homography is supported for now.
    RH 2022

    Args:
        im_template (np.ndarray):
            Template image
        im_moving (np.ndarray):
            Moving image
        warp_mode (str):
            warp mode.
            See cv2.findTransformECC for more info.
            MOTION_TRANSLATION sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
            MOTION_EUCLIDEAN sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
            MOTION_AFFINE sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
            MOTION_HOMOGRAPHY sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
        n_iter (int):
            Number of iterations
        termination_eps (float):
            Termination epsilon.
            Threshold of the increment in the correlation
             coefficient between two iterations
        mask (np.ndarray):
            Binary mask. If None, no mask is used.
            Regions where mask is zero are ignored 
             during the registration.
        gaussFiltSize (int):
            gaussian filter size. If 0, no gaussian 
             filter is used.
    
    Returns:
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
            Can be applied using cv2.warpAffine or 
             cv2.warpPerspective.
    """
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        termination_eps,
    )
    print("running findTransformECC")
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        templateImage=im_template, 
        inputImage=im_moving, 
        warpMatrix=warp_matrix,
        motionType=warp_mode, 
        criteria=criteria, 
        inputMask=mask, 
        gaussFiltSize=gaussFiltSize
    )
    return warp_matrix


def apply_warp_transform(
    im_in,
    warp_matrix,
    interpolation_method=cv2.INTER_LINEAR, 
    borderMode=cv2.BORDER_CONSTANT, 
    borderValue=0
):
    """
    Apply a warp transform to an image.
    RH 2022

    Args:
        im_in (np.ndarray):
            Input image
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
        interpolation_method (int):
            Interpolation method.
            See cv2.warpAffine for more info.
        borderMode (int):
            Border mode.
            Whether to use a constant border value or not.
            See cv2.warpAffine for more info.
        borderValue (int):
            Border value.

    Returns:
        im_out (np.ndarray):
            Output image
    """
    im_out = cv2.warpPerspective(
        src=im_in,
        M=warp_matrix,
        dsize=(im_in.shape[1], im_in.shape[0]), 
        dst=copy.copy(im_in), 
        flags=interpolation_method, 
        borderMode=borderMode, 
        borderValue=borderValue
    )
    return im_out

def stack_to_RGB(images):
    """
    Convert a stack of images to RGB.
    RH 2022

    Args:
        images (list of np.ndarray):
            List of images.
            Can be between 1 and 3 images.
            Can also be a single image.

    Returns:
        im_out (np.ndarray):
            RGB image.
    """
    if isinstance(images, np.ndarray):
        images = [images]
    
    im_out = np.stack(images, axis=2)
    appended_images = np.stack([images[0]*0] * (3 - len(images)), axis=2)
    im_out = np.concatenate([im_out, appended_images], axis=2)

    return im_out

def bin_array(array, bin_widths=[2,3,4], method='append', function=np.nanmean, function_kwargs={}):
    """
    Bins an array of arbitrary shape along the
     first N dimensions. Works great for images.
    Works by iteratively reshaping and applying
     the defined function (ie averaging).
    Function requires pad_with_singleton_dims function.
    RH 2022

    Args:
        array (np.ndarray):
            Input array.
        bin_widths (list of int):
            List of bin widths for first N dimensions.
        method (str):
            Method for binning.
            'append' appends NaNs to the end of
             each dimension that is not divisible
             by the bin width.
            'prepend' prepends NaNs to the beginning
             of each dimension that is not divisible
             by the bin width.
            'post_crop' crops the array to be divisible
             by the bin width by cropping off the end.
            'pre_crop' crops the array to be divisible
             by the bin width by cropping off the 
             beginning.
        function (function):
            Function to apply to each bin.
            Must at least take array and axis as arguments.
            Doesn't need to handle NaNs if 
             method==pre or post_crop.
            Typically: 
                np.nanmean
                np.nanmedian
                np.nanstd
                np.nanvar
                np.nanmax
                np.nanmin
                np.nanpercentile (requires percentile kwarg)
                np.nanquantile (requires percentile kwarg)
        function_kwargs (dict):
            Keyword arguments for function.

    Returns:
        array_out (np.ndarray):
            Output array.
    """
    
    # A cute function for flattening a list of lists.
    flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

    s = list(array.shape)

    arr_out = copy.copy(array)
    for n, w in enumerate(bin_widths):
        if arr_out.shape[n] % w != 0:
            if method=='append':
                s_pad = copy.copy(s)
                s_pad[n] = w - (arr_out.shape[n] % w)
                arr_out = np.concatenate(
                    [arr_out, np.zeros(s_pad)*np.nan],
                    axis=n
                )
            
            if method=='prepend':
                s_pad = copy.copy(s)
                s_pad[n] = w - (arr_out.shape[n] % w)
                arr_out = np.concatenate(
                    [np.zeros(s_pad)*np.nan, arr_out],
                    axis=n
                )
                
            if method=='post_crop':
                s_crop = indexing.pad_with_singleton_dims(np.arange((arr_out.shape[n]//w)*w), n_dims_pre=n, n_dims_post=array.ndim-(n+1))
                arr_out = np.take_along_axis(arr_out, s_crop, axis=n)

            if method=='pre_crop':
                s_crop = indexing.pad_with_singleton_dims(np.arange((arr_out.shape[n]//w)*w) + arr_out.shape[n]%w, n_dims_pre=n, n_dims_post=array.ndim-(n+1))
                arr_out = np.take_along_axis(arr_out, s_crop, axis=n)
        
        s_n = list(arr_out.shape)
        s_n[n] = [w, arr_out.shape[n] // w]
        s_n = flatten_list(s_n)
        arr_out = np.reshape(arr_out, s_n, order='F')
        arr_out = function(arr_out, axis=n, **function_kwargs)
        
    return arr_out


def flatten_channels_along_frames(images):
    """
    Concatenates channels along frame dimension.
    Opposite of reravel_channels.
    RH 2022

    Args:
        images: 
            Input images or video.
            shape(frames, height, width, channels)
    
    Returns:
        output images:
            Reshaped images.
            shape(frames*channels, height, width)
    """
    return images.transpose(1,2,0,3).reshape(images.shape[1], images.shape[2], images.shape[0]*images.shape[3]).transpose(2,0,1)

def reravel_channels(images, n_channels=3):
    """
    De-concatenates channels from the frame dimension to a new dimension.
    Opposite of flatten_channels_along_frames.
    RH 2022
    
    Args:
        images:
            Input images or video. Channels should be concatenated in an
             interleaved fashion along the frame dimension.
            shape(frames*channels, height, width)
        n_channels:
            Number of channels.

    Returns:
        output images:
            Reshaped images.
            shape(frames, height, width, channels)
    """
    return images.transpose(1,2,0).reshape(images.shape[1], images.shape[2], images.shape[0]//n_channels, n_channels).transpose(2,0,1,3)


def center_crop_images(images, height_width=None):
    """
    Crops images around the center.
    RH 2022

    Args:
        images:
            Input images or video.
            shape(frames, height, width, channels)
        height_width:
            2-tuple or list of height and width to crop to.
            If None, then set to the minimum of the height or width
             of the images (making the images square).

    Returns:
        output images:
            Cropped images.
    """
    if height_width is None:
        minimum_shape = min(images.shape[1], images.shape[2])
        height = minimum_shape
        width  = minimum_shape
    else:
        height = height_width[0]
        width  = height_width[1]
        
    center_idx = [images.shape[1]//2, images.shape[2]//2]
    height_half = height//2
    width_half  = width//2
    
    return images[:, center_idx[0] - height_half : center_idx[0] + height_half, center_idx[1] - width_half : center_idx[1] + width_half, :]

def center_pad_images(images, height_width=None):
    """
    Pads images around the center.
    RH 2022

    Args:
        images:
            Input images or video.
            shape(frames, height, width, channels)
        height_width:
            2-tuple or list of height and width to pad to.
            If None, then set to the maximum of the height or width
             of the images (making the images square).

    Returns:
        output images:
            Padded images.
    """
    if height_width is None:
        maximum_shape = max(images.shape[1], images.shape[2])
        height = maximum_shape
        width = maximum_shape
    else:
        height = height_width[0]
        width = height_width[1]
        
    pad_height = (height - images.shape[1]) // 2
    pad_width  = (width - images.shape[2]) // 2
    images_out = np.zeros((images.shape[0], height, width, images.shape[3]))

    images_out[:, pad_height : pad_height + images.shape[1], pad_width : pad_width + images.shape[2], :] = images
    
    return images_out


def make_tiled_video_array(
    paths_videos, 
    frame_idx_list, 
    block_height_width=[300,300],
    n_channels=3, 
    tiling_shape=None, 
    dtype=np.uint8,
    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
    crop_idx=None,
    overlay_signals=None,
    overlay_idx=None,
    spacer_black_frames=0,
    pixel_val_range=None,
    ):
    """
    Creates a tiled video array from a list of paths to videos.
    NOTE: On my Ubuntu machine:
        - importing 'av' after cv2 causes cv2.imshow to hang forever and
        - importing 'decord' after cv2 causes cv2.imshow to crash the kernel
    RH 2022

    Args:
        paths_videos:
            List of paths to videos.
        frame_idx_list (ndarray, 3D, int):
            Shape: (n_chunks, 2, n_videos)
            Second dimension: (start_frame, end_frame) of chunk.
            Values should be positive integers.
            To insert black frames instead of video chunk, use
             [-1, -1] for the idx tuple.
        block_height_width:
            2-tuple or list of height and width of each block.
        n_channels:
            Number of channels.
        tiling_shape:
            2-tuple or list of height and width of the tiled video.
            If None, then set to be square and large enough to 
             contain all the blocks.
        dtype:
            Data type of the output array. Should match the data
             type of the input videos.
        interpolation:
            Interpolation mode for the video. Should be one of the
             torchvision.transforms.InterpolationMode values.
        crop_idx:
            List of 4-tuples or lists of indices to crop the video.
            [top, bottom, left, right]
            If None, then no cropping is performed.
            Outer list should be same length as paths_videos. Each 
             entry should correspond to the crop indices for a video.
        overlay_signals:
            List of signals to overlay on the video.
            Each signal should be a numpy array of shape(frames, n_channels).
            The signals will be represented as a white rectangle with
             indices from overlay_idx.
        overlay_idx:
            List of indices to overlay the signals.
            [top, bottom, left, right]
        spacer_black_frames:
            Number of black frames to add between each chunk.
        pixel_val_range:
            2-tuple or list of the minimum and maximum pixel values.
            If None, then no clipping is performed.

    Returns:
        output video array:
            Tiled video array.
            shape(frames, tiling_shape[0]*block_height_width[0], tiling_shape[1]*block_height_width[1], channels)
    """

    ##  Example values
    ##  - frame_idx_list = [np.array([[703,843], [743,883], [799, 939], [744, 884]]*2).T, np.array([[39,89], [43,93], [99, 149], [44, 94]]*2).T]
    ##  - frame_idx_list = [np.array([[37900,38050], [37900,38050], [37900,38050], [37900,38050]]*2).T, np.array([[37900,38050], [37900,38050], [37900,38050], [37900,38050]]*2).T]
    
    ##  - roi = plotting_helpers.select_ROI(image)
    ##  - pts = np.array(roi.selected_points).squeeze().astype('uint32')
    ##  - crop_idx = [pts[0,1], pts[1,1], pts[0,0], pts[1,0]]*8
    ##  - block_height_width=[crop_idx[1]-crop_idx[0] , crop_idx[3]-crop_idx[2]]
    ##  - tiling_shape = [2,2]
    ##  - spacer_black_frames = 5
    ##  - paths_videos = [path_video] * 8


    def resize_torch(images, new_shape=[100,100], interpolation=interpolation):
        resize = torchvision.transforms.Resize(new_shape, interpolation=interpolation, max_size=None, antialias=None)
        return resize(torch.as_tensor(images)).numpy()

    def add_overlay(chunk, overlay_signal, overlay_idx):
        ol_height = overlay_idx[1]-overlay_idx[0]
        ol_width = overlay_idx[3]-overlay_idx[2]
        ol = np.ones((chunk.shape[0], ol_height, ol_width, chunk.shape[3])) * overlay_signal[:,None,None,None]
        chunk[:, overlay_idx[0]:overlay_idx[1], overlay_idx[2]:overlay_idx[3], :] = ol
        return chunk

    duration_chunks = frame_idx_list[:,1,:] - frame_idx_list[:,0,:] + spacer_black_frames
    max_frames_per_chunk = np.nanmax(duration_chunks, axis=1)

    null_chunks = (frame_idx_list == -1).all(axis=1)
    
    ## ASSERTIONS
    ## check to make sure that shapes are correct
    for i_chunk, chunk in enumerate(frame_idx_list):
        assert chunk.shape[0] == 2, f'RH ERROR: size of first dimension of each frame_idx matrix should be 2'
        assert chunk.shape[1] == len(paths_videos), f'RH ERROR: size of second dimension of each frame_idx matrix should match len(paths_videos)'

    n_vids = len(paths_videos)
    n_frames_total = max_frames_per_chunk.sum()  ## total number of frames in the final video
    block_aspect_ratio = block_height_width[0] / block_height_width[1]

    cum_start_idx_chunk = np.cumsum(np.concatenate(([0], max_frames_per_chunk)))[:-1] ## cumulative starting indices of temporal chunks in final video

    if tiling_shape is None:
        el = int(np.ceil(np.sqrt(n_vids)))  ## 'edge length' in number of videos
        tiling_shape = [el, el]  ## n_vids high , n_vids wide
    tile_grid_tmp = np.meshgrid(np.arange(tiling_shape[0]), np.arange(tiling_shape[1]))
    tile_position_vids = [np.reshape(val, -1, 'F') for val in tile_grid_tmp]  ## indices of tile/block positions for each video

    vid_height_width = list(np.array(block_height_width) * tiling_shape)  ## total height and width of final video


    tile_topLeft_idx = [[tile_position_vids[0][i_vid]*block_height_width[0], tile_position_vids[1][i_vid]*block_height_width[1]] for i_vid in range(len(paths_videos))]  ## indices of the top left pixels for each tile/block. List of lists: outer list is tile/block, inner list is [y,x] starting idx

    video_out = np.zeros((n_frames_total, vid_height_width[0], vid_height_width[1], n_channels), dtype)  ## pre-allocation of final video array

    for i_vid, path_vid in enumerate(tqdm(paths_videos)):
        if isinstance(path_vid, list):
            flag_multivid = True
            multivid_lens = [av.open(str(path)).streams.video[0].frames for path in path_vid]
            # multivid_lens = [len(decord.VideoReader(path)) for path in path_vid]  ## decord method of same thing
            cum_start_idx_multiVid = np.cumsum(np.concatenate(([0], multivid_lens)))[:-1]
        else:
            flag_multivid = False


        for i_chunk, idx_chunk in enumerate(frame_idx_list):
            if null_chunks[i_chunk, i_vid]:
                continue
            elif flag_multivid:
                frames_remainder = idx_chunk[1,i_vid] - idx_chunk[0,i_vid]  ## initialization of remaining frames
                frame_toStartGlobal = idx_chunk[0,i_vid]  ## frame to start at (in concatenated frame indices)

                chunks_list = []
                while frames_remainder > 0:
                    multivid_toStart = indexing.get_last_True_idx((frame_toStartGlobal - cum_start_idx_multiVid) >= 0)  ## which of the multivids to start at

                    frame_toStartInVid = frame_toStartGlobal - cum_start_idx_multiVid[multivid_toStart]  ## where to start in the vid

                    frames_toEndOfVid = multivid_lens[multivid_toStart] - frame_toStartInVid  ## number of frames left in the vid
                    frames_toGrab = min(frames_remainder  ,  frames_toEndOfVid)  ## number of frames to get from current vid
                    frames_remainder -= frames_toGrab

                    vid = decord.VideoReader(str(path_vid[multivid_toStart]), ctx=decord.cpu())  ## open the vid
                    chunks_list.append(vid[frame_toStartInVid : frame_toStartInVid+frames_toGrab].asnumpy())  ## raw video chunk
                    frame_toStartGlobal += frames_toGrab

                chunk = np.concatenate(chunks_list, axis=0)
            else:
                vid = decord.VideoReader(path_vid, ctx=decord.cpu())
                chunk = vid[idx_chunk[0, i_vid] : idx_chunk[1, i_vid]].asnumpy()  ## raw video chunk

            chunk_height, chunk_width, chunk_n_frames, _ = chunk.shape
            if crop_idx is not None:
                chunk = chunk[:, crop_idx[i_vid][0]:crop_idx[i_vid][1], crop_idx[i_vid][2]:crop_idx[i_vid][3], :]

            ## first we get the aspect ratio right by padding to correct aspect ratio
            aspect_ratio = chunk.shape[1] / chunk.shape[2]
            if aspect_ratio >= block_aspect_ratio:
                tmp_height = chunk.shape[1]
                tmp_width = int(np.ceil(chunk.shape[1] / block_aspect_ratio))
            if aspect_ratio < block_aspect_ratio:
                tmp_height = int(np.ceil(chunk.shape[2] * block_aspect_ratio))
                tmp_width = chunk.shape[2]
            chunk_ar = center_pad_images(chunk, height_width=[tmp_height, tmp_width])

            ## then we resize the movie to the final correct size
            chunk_rs = resize_torch(chunk_ar.transpose(0,3,1,2), new_shape=block_height_width, interpolation=interpolation).transpose(0,2,3,1)

            if pixel_val_range is not None:
                chunk_rs[chunk_rs < pixel_val_range[0]] = pixel_val_range[0]  ## clean up interpolation errors
                chunk_rs[chunk_rs > pixel_val_range[1]] = pixel_val_range[1]

            ## add overlay to the chunk
            if overlay_signals is not None:
                add_overlay(chunk_rs, overlay_signals[i_vid][idx_chunk[0,i_vid]:idx_chunk[1,i_vid], i_chunk], overlay_idx)


            ## drop into final video array
            video_out[
                cum_start_idx_chunk[i_chunk] : duration_chunks[i_chunk, i_vid] + cum_start_idx_chunk[i_chunk] - spacer_black_frames,
                tile_topLeft_idx[i_vid][0] : tile_topLeft_idx[i_vid][0]+block_height_width[0], 
                tile_topLeft_idx[i_vid][1] : tile_topLeft_idx[i_vid][1]+block_height_width[1], 
                :
            ] = chunk_rs

    return video_out


def add_text_to_images(images, text, position=(10,10), font_size=1, color=(255,255,255), line_width=1, font=None, show=False, frameRate=30):
    """
    Add text to images using cv2.putText()
    RH 2022

    Args:
        images (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        text (list of lists):
            text to add to images.
            Outer list: one element per frame.
            Inner list: each element is a line of text.
        position (tuple):
            (x,y) position of text (top left corner)
        font_size (int):
            font size of text
        color (tuple):
            (r,g,b) color of text
        line_width (int):
            line width of text
        font (str):
            font to use.
            If None, then will use cv2.FONT_HERSHEY_SIMPLEX
            See cv2.FONT... for more options
        show (bool):
            if True, then will show the images with text added.

    Returns:
        images_with_text (np.array):
            frames of video or images with text added.
    """
    import cv2
    import copy
    
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    images_cp = copy.copy(images)
    for i_f, frame in enumerate(images_cp):
        for i_t, t in enumerate(text[i_f]):
            cv2.putText(frame, t, [position[0] , position[1] + i_t*font_size*30], font, font_size, color, line_width)
        if show:
            cv2.imshow('add_text_to_images', frame)
            cv2.waitKey(int(1000/frameRate))
    
    if show:
        cv2.destroyWindow('add_text_to_images')
    return images_cp


def add_image_overlay(
    images_overlay,
    images_underlay,
    position_topLeft=(0,0),
    height_width=(100,100),
    interpolation=None,
):
    """
    Add an image/video overlay to an image/video underlay.
    RH 2022

    Args:
        images_overlay (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        images_underlay (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        position_topLeft (tuple):
            (y,x) position of overlay (top left corner of overlay)
        height_width (tuple):
            (height, width) of overlay
            Images will be resized to this size.
        interpolation (str):
            interpolation method to use.
            
    Returns:
        images_with_overlay (np.array):
            frames of video or images with overlay added.
    """
    
    if interpolation is None:
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        
    def resize_torch(images, new_shape=[100,100], interpolation=interpolation):
        resize = torchvision.transforms.Resize(new_shape, interpolation=interpolation, max_size=None, antialias=None)
        return resize(torch.as_tensor(images)).numpy()

    im_hw = images_underlay[0].shape

    pos_all = np.array([position_topLeft[0], position_topLeft[0]+height_width[0], position_topLeft[1], position_topLeft[1]+height_width[1]])
    pos_all[pos_all < 0] = 0
    pos_all[1] = min(im_hw[0], pos_all[1])
    pos_all[3] = min(im_hw[1], pos_all[3])

    rs_size = [pos_all[1]-pos_all[0], pos_all[3]-pos_all[2]]

    images_rs = resize_torch(images_overlay.transpose(0,3,1,2), new_shape=rs_size, interpolation=interpolation).transpose(0,2,3,1)

    images_out = copy.copy(images_underlay)
    images_out[:, pos_all[0]:pos_all[1], pos_all[2]:pos_all[3],:] = images_rs
    
    return images_out


    #############################################
################ Video ######################
#############################################

def play_video_cv2(array=None, path=None, frameRate=30, save_path=None, show=True, fourcc_code='MJPG', text=None, kwargs_text={}):
    """
    Play a video using OpenCV
    RH 2021

    Args:
        array:
            Either 3D array of images (frames x height x width)
             or a 4D array of images (frames x height x width x channels)
            Scaling assumed to be between 0 and 255
            If None, then path must be specified
        path:
            Path to video file
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
        size = tuple((np.flip(array.shape[1:3])))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        print(f'saving to file {save_path}')
        writer = cv2.VideoWriter(save_path, fourcc, frameRate, size)

    if kwargs_text is None:
        kwargs_text = { 'org': (5, 15), 
                        'fontFace': 1, 
                        'fontScale': 1,
                        'color': (255, 255, 255), 
                        'thickness': 1}
    
    if array is not None:

        array[array < 0] = 0
        array[array > 255] = 255
        if array.dtype != 'uint8':
            array = array.astype('uint8')
        movie = array
        if array.ndim == 4:
            flag_convert_to_gray = True
        else:
            flag_convert_to_gray = False
    else:
        movie = decord.VideoReader(path)
        flag_convert_to_gray = False

    for i_frame, frame in enumerate(tqdm(movie)):
        frame = frame.asnumpy()

        if array is not None:
            if flag_convert_to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                Exception('RH: Unsupported number of channels, check array shape')
            # else:  
            #     frame = cv2.merge([frame, frame, frame])

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

