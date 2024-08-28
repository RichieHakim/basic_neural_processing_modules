import threading
from typing import Union, List, Tuple, Optional
import time
from pathlib import Path

import torch
import torchvision
import numpy as np
import cv2
from tqdm.auto import tqdm


###############################################################################
## This block of code is used to initialize cv2.imshow
## This is necessary because importing av and decord 
##  will cause cv2.imshow to fail unless it is initialized.
## Obviously, this should be commented out when running on
##  systems that do not support cv2.imshow like servers.
## Also be sure to import BNPM before importing most other
##  modules.
def prepare_cv2_imshow():
    """
    This function is necessary because cv2.imshow() 
     can crash the kernel if called after importing 
     av and decord.
    RH 2022
    """
    import numpy as np
    import cv2
    test = np.zeros((1,300,400,3))
    for frame in test:
        cv2.putText(frame, "WELCOME TO BNPM!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Prepping CV2", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to work without crashing if this function", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "is called before importing av and decord", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(100)
    cv2.destroyWindow('startup')
###############################################################################


# import av
import decord
# # import cv2 

from . import indexing, image_processing

def prepare_cv2_imshow():
    test = np.zeros((1,300,400,3))
    for frame in test:
        cv2.putText(frame, "Prepping CV2", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to run after importing av and decord", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(100)
    cv2.destroyWindow('startup')

def play_video_cv2(
    array=None, 
    path_video=None, 
    frameRate=30, 
    path_save=None, 
    show=True, 
    fourcc_code='MJPG', 
    text=None, 
    kwargs_text={}
):
    """
    Play a video using OpenCV
    RH 2021

    Args:
        array:
            Either 3D array of images (frames x height x width)
             or a 4D array of images (frames x height x width x channels)
            Scaling assumed to be between 0 and 255
            If None, then path must be specified
        path_video:
            Path to video file
            If None, then array must be specified
        frameRate:  
            Frame rate of the video (in Hz)
        path_save:
            Path to save the video.
            If None, then video is not saved.
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
    import decord
    wait_frames = max(int((1/frameRate)*1000), 1)
    if path_save is not None:
        size = tuple((np.flip(array.shape[1:3])))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        print(f'saving to file {path_save}')
        writer = cv2.VideoWriter(path_save, fourcc, frameRate, size)

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
        elif array.ndim == 3:
            flag_convert_to_gray = False
        else:
            raise Exception('RH: Unsupported number of channels, check array shape')
    else:
        movie = decord.VideoReader(path_video)
        flag_convert_to_gray = False

    for i_frame, frame in enumerate(tqdm(movie)):
        if array is None:
            frame = frame.asnumpy()

        if array is not None:
            if flag_convert_to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
        if path_save is not None:
            writer.write(np.uint8(frame))
            # print(flag_convert_to_gray, frame.shape, frame.dtype)
    if path_save is not None:
        writer.release()
        print('Video saved')
    if show:
        cv2.destroyWindow('handle')


def play_video_pygame(
    array=None,
    frameRate=30,
    loop=True,
    frame_counter=False,
):
    """
    Play a video using Pygame.
    Press 'q' or 'ESC' to quit.
    RH 2024

    Args:
        array (np.ndarray):
            3D or 4D array of images. Shape should be (frames, height, width) or
            (frames, height, width, channels).
            dtype must be uint8, and scaling assumed to be between 0 and 255.
        frameRate:  
            Frame rate of the video (in Hz)
        loop (bool):
            Whether to loop the video or not.
        frame_counter (bool):
            Display a frame counter (zero-indexed).
    """
    import pygame
    
    if array.ndim == 3:
        array = np.stack([array]*3, axis=-1)

    array = array.transpose(0, 2, 1, 3)
    frame_height, frame_width = array.shape[2], array.shape[1]
    
    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    clock = pygame.time.Clock()

    running = True
    i_frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

        screen.blit(pygame.surfarray.make_surface(array[i_frame]), (0, 0))
        if frame_counter:
            font = pygame.font.Font(None, 36)
            text = font.render(f'{i_frame}', True, (255, 255, 255))
            screen.blit(text, (10, 10))        
        pygame.display.flip()
        clock.tick(frameRate)
        if not loop:
            if i_frame >= array.shape[0] - 1:
                running = False
        i_frame = (i_frame + 1) % array.shape[0]
        
    pygame.quit()


def grayscale_to_rgb(array):
    """
    Convert a grayscale image (2D array) or movie (3D array) to
     RGB (3D or 4D array).
    RH 2023

    Args:
        array (np.ndarray or torch.Tensor or list):
            2D or 3D array of grayscale images
    """
    if isinstance(array, list):
        if isinstance(array[0], np.ndarray):
            array = np.stack(array, axis=0)
        elif isinstance(array[0], torch.Tensor):
            array = torch.stack(array, axis=0)
        else:
            raise Exception(f'Failed to convert list of type {type(array[0])} to array')
    if isinstance(array, np.ndarray):
        return np.stack([array, array, array], axis=-1)
    elif isinstance(array, torch.Tensor):
        return torch.stack([array, array, array], dim=-1)
    

def save_gif(
    array, 
    path, 
    frameRate=5.0, 
    loop=0, 
    backend='PIL', 
    kwargs_backend={},
):
    """
    Save an array of images as a gif.
    RH 2023

    Args:
        array (np.ndarray or list):
            3D (grayscale) or 4D (color) array of images.
            - if dtype is float type then scale from 0 to 1.
            - if dtype is integer then scale from 0 to 255.
        path (str):
            Path to save the gif.
        frameRate (float):
            Frame rate of the gif.
        loop (int):
            Number of times to loop the gif.
            0 mean loop forever
            1 mean play once
            2 means play twice (loop once)
            etc.
        backend (str):
            Which backend to use.
            Options: 'imageio' or 'PIL'
        kwargs_backend (dict):
            Keyword arguments for the backend.
    """
    array = np.stack(array, axis=0) if isinstance(array, list) else array
    array = grayscale_to_rgb(array) if array.ndim == 3 else array
    if np.issubdtype(array.dtype, np.floating):
        array = (array*255).astype('uint8')
    
    kwargs_backend.update({'loop': loop} if loop != 1 else {})

    if backend == 'imageio':
        import imageio
        imageio.mimsave(
            path, 
            array, 
            format='GIF',
            duration=1000/frameRate, 
            **kwargs_backend,
        )
    elif backend == 'PIL':
        from PIL import Image
        frames = [Image.fromarray(array[i_frame]) for i_frame in range(array.shape[0])]
        frames[0].save(
            path, 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True, 
            duration=1000/frameRate, 
            **kwargs_backend,
        )
    else:
        raise Exception(f'Unsupported backend {backend}')


class VideoReaderWrapper(decord.VideoReader):
    """
    Used to fix a memory leak bug in decord.VideoReader
    Taken from here.
    https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seek(0)
        
        self.path = args[0]

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames


class BufferedVideoReader:
    """
    A video reader that loads chunks of frames into a memory buffer
     in background processes so that sequential batches of frames
     can be accessed quickly.
    In many cases, allows for reading videos in batches without
     waiting for loading of the next batch to finish.
    Uses threading to read frames in the background.

    Optimal use case:
    1. Create a _BufferedVideoReader object
    EITHER 2A. Set method_getitem to 'continuous' and iterate over the
        object. This will read frames continuously in the
        background. This is the fastest way to read frames.
    OR 2B. Call batches of frames sequentially. Going backwards is
        slow. Buffers move forward.
    3. Each batch should be within a buffer. There should be no
        batch slices that overlap multiple buffers. Eg. if the
        buffer size is 1000 frames, then the following is fast:
        [0:1000], [1000:2000], [2000:3000], etc.
        But the following are slow:
        [0:1700],  [1700:3200],   [0:990],         [990:1010], etc.
        ^too big,  ^2x overlaps,  ^went backward,  ^1x overlap

    RH 2022
    """
    def __init__(
        self,
        video_readers: list=None,
        paths_videos: list=None,
        buffer_size: int=1000,
        prefetch: int=2,
        posthold: int=1,
        method_getitem: str='continuous',
        starting_seek_position: int=0,
        decord_backend: str='torch',
        decord_ctx=None,
        verbose: int=1,
    ):
        """
        video_readers (list of decord.VideoReader): 
            list of decord.VideoReader objects.
            Can also be single decord.VideoReader object.
            If None, then paths_videos must be provided.
        paths_videos (list of str):
            list of paths to videos.
            Can also be single str.
            If None, then video_readers must be provided.
            If both paths_videos and video_readers are provided, 
             then video_readers will be used.
        buffer_size (int):
            Number of frames per buffer slot.
            When indexing this object, try to not index more than
             buffer_size frames at a time, and try to not index
             across buffer slots (eg. across idx%buffer_size==0).
             These require concatenating buffers, which is slow.
        prefetch (int):
            Number of buffers to prefetch.
            If 0, then no prefetching.
            Note that a single buffer slot can only contain frames
             from a single video. Best to keep 
             buffer_size <= video length.
        posthold (int):
            Number of buffers to hold after a new buffer is loaded.
            If 0, then no posthold.
            This is useful if you want to go backwards in the video.
        method_getitem (str):
            Method to use for __getitem__.
            'continuous' - read frames continuously across videos.
                Index behaves like videos are concatenated:
                - reader[idx] where idx: slice=idx_frames
            'by_video' - index must specify video index and frame 
                index:
                - reader[idx] where idx: tuple=(int: idx_video, slice: idx_frames)
        starting_seek_position (int):
            Starting frame index to start iterator from.
            Only used when method_getitem=='continuous' and
             using the iterator method.
        decord_backend (str):
            Backend to use for decord when loading frames.
            See decord documentation for options.
            ('torch', 'numpy', 'mxnet', ...)
        decord_ctx (decord.Context):
            Context to use for decord when loading frames.
            See decord documentation for options.
            (decord.cpu(), decord.gpu(), ...)
        verbose (int):
            Verbosity level.
            0: no output
            1: output warnings
            2: output warnings and info
        """
        import pandas as pd

        self._verbose = verbose
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.posthold = posthold
        self._decord_backend = decord_backend
        self._decord_ctx = decord.cpu(0) if decord_ctx is None else decord_ctx

        ## Check inputs
        if isinstance(video_readers, decord.VideoReader):
            video_readers = [video_readers]
        if isinstance(paths_videos, str):
            paths_videos = [paths_videos]
        assert (video_readers is not None) or (paths_videos is not None), "Must provide either video_readers or paths_videos"

        ## If both video_readers and paths_videos are provided, use the video_readers and print a warning
        if (video_readers is not None) and (paths_videos is not None):
            print(f"FR WARNING: Both video_readers and paths_videos were provided. Using video_readers and ignoring path_videos.")
            paths_videos = None
        ## If paths are specified, import them as decord.VideoReader objects
        if paths_videos is not None:
            print(f"FR: Loading lazy video reader objects...") if self._verbose > 1 else None
            assert isinstance(paths_videos, list), "paths_videos must be list of str"
            assert all([isinstance(p, str) for p in paths_videos]), "paths_videos must be list of str"
            video_readers = [VideoReaderWrapper(path_video, ctx=self._decord_ctx) for path_video in tqdm(paths_videos, disable=(self._verbose < 2))]
            self.paths_videos = paths_videos
        else:
            print(f"FR: Using provided video reader objects...") if self._verbose > 1 else None
            assert isinstance(video_readers, list), "video_readers must be list of decord.VideoReader objects"
            self.paths_videos = [v.path for v in video_readers]
            assert all([isinstance(v, decord.VideoReader) for v in video_readers]), "video_readers must be list of decord.VideoReader objects"
        ## Assert that method_getitem is valid
        assert method_getitem in ['continuous', 'by_video'], "method_getitem must be 'continuous' or 'by_video'"
        ## Check if backend is valid by trying to set it here (only works fully when used in the _load_frames method)
        decord.bridge.set_bridge(self._decord_backend)

        self.paths_videos = [str(path) for path in self.paths_videos]  ## ensure paths are str
        self.video_readers = video_readers
        self._cumulative_frame_end = np.cumsum([len(video_reader) for video_reader in self.video_readers])
        self._cumulative_frame_start = np.concatenate([[0], self._cumulative_frame_end[:-1]])
        self.num_frames_total = self._cumulative_frame_end[-1]
        self.method_getitem = method_getitem

        ## Get metadata about videos: lengths, fps, frame size, etc.
        self.metadata, self.num_frames_total, self.frame_rate, self.frame_height_width, self.num_channels = self._get_metadata(self.video_readers)
        ## Get number of videos
        self.num_videos = len(self.video_readers)

        ## Set shape
        self.shape = (self.num_frames_total, *self.frame_height_width, self.num_channels)

        ## Set iterator starting frame
        print(f"FR: Setting iterator starting frame to {starting_seek_position}") if self._verbose > 1 else None
        self.set_iterator_frame_idx(starting_seek_position)

        ## Initialize the buffer
        ### Make a list containing a slot for each buffer chunk
        self.slots = [[None] * np.ceil(len(d)/self.buffer_size).astype(int) for d in self.video_readers]
        ### Make a list containing the bounding indices for each buffer video chunk. Upper bound should be min(buffer_size, num_frames)
        self.boundaries = [[(i*self.buffer_size, min((i+1)*self.buffer_size, len(d))-1) for i in range(len(s))] for d, s in zip(self.video_readers, self.slots)]
        ### Make a lookup table for the buffer slot that contains each frame
        self.lookup = {
            'video': np.concatenate([np.array([ii]*len(s), dtype=int) for ii, s in enumerate(self.slots)]).tolist(),
            'slot': np.concatenate([np.arange(len(s)) for s in self.slots]).tolist(),
            'start_frame': np.concatenate([np.array([s[0] for s in b]) for b in self.boundaries]).astype(int).tolist(), 
            'end_frame': np.concatenate([np.array([s[1] for s in b]) for b in self.boundaries]).astype(int).tolist(),
        }
        self.lookup['start_frame_continuous'] = (np.array(self.lookup['start_frame']) + np.array(self._cumulative_frame_start[self.lookup['video']])).tolist()
        self.lookup = pd.DataFrame(self.lookup)
        self._start_frame_continuous = self.lookup['start_frame_continuous'].values

        ## Make a list for which slots are loaded or loading
        self.loading = []
        self.loaded = []


    def _get_metadata(self, video_readers):
        """
        Get metadata about videos: lengths, fps, frame size, 
         num_channels, etc.

        Args:
            video_readers (list of decord.VideoReader):
                List of decord.VideoReader objects

        Returns:
            metadata (list of dict):
                Dictionary containing metadata for each video.
                Contains: 'num_frames', 'frame_rate',
                 'frame_height_width', 'num_channels'
            num_frames_total (int):
                Total number of frames across all videos.
            frame_rate (float):
                Frame rate of videos.
            frame_height_width (tuple of int):
                Height and width of frames.
            num_channels (int):
                Number of channels.
        """

        ## make video metadata dataframe
        print("FR: Collecting video metadata...") if self._verbose > 1 else None
        metadata = {"paths_videos": self.paths_videos}
        num_frames, frame_rate, frame_height_width, num_channels = [], [], [], []
        for v in tqdm(video_readers, disable=(self._verbose < 2)):
            num_frames.append(int(len(v)))
            frame_rate.append(float(v.get_avg_fps()))
            frame_tmp = v[0]
            frame_height_width.append([int(n) for n in frame_tmp.shape[:2]])
            num_channels.append(int(frame_tmp.shape[2]))
        metadata["num_frames"] = num_frames
        metadata["frame_rate"] = frame_rate
        metadata["frame_height_width"] = frame_height_width
        metadata["num_channels"] = num_channels
            

        ## Assert that all videos must have at least one frame
        assert all([n > 0 for n in metadata["num_frames"]]), "FR ERROR: All videos must have at least one frame"
        ## Assert that all videos must have the same shape
        assert all([n == metadata["frame_height_width"][0] for n in metadata["frame_height_width"]]), "FR ERROR: All videos must have the same shape"
        ## Assert that all videos must have the same number of channels
        assert all([n == metadata["num_channels"][0] for n in metadata["num_channels"]]), "FR ERROR: All videos must have the same number of channels"

        ## get frame rate
        frame_rates = metadata["frame_rate"]
        ## warn if any video's frame rate is very different from others
        max_diff = float((np.max(frame_rates) - np.min(frame_rates)) / np.mean(frame_rates))
        print(f"FR WARNING: max frame rate difference is large: {max_diff*100:.2f}%") if ((max_diff > 0.1) and (self._verbose > 0)) else None
        frame_rate = float(np.median(frame_rates))

        num_frames_total = int(np.sum(metadata["num_frames"]))
        frame_height_width = metadata["frame_height_width"][0]
        num_channels = metadata["num_channels"][0]

        return metadata, num_frames_total, frame_rate, frame_height_width, num_channels


    def _load_slots(self, idx_slots: list, wait_for_load: Union[bool, list]=False):
        """
        Load slots in the background using threading.

        Args:
            idx_slots (list): 
                List of tuples containing the indices of the slots to load.
                Each tuple should be of the form (idx_video, idx_buffer).
            wait_for_load (bool or list):
                If True, wait for the slots to load before returning.
                If False, return immediately.
                If True wait for each slot to load before returning.
                If a list of bools, each bool corresponds to a slot in
                 idx_slots.
        """
        ## Check if idx_slots is a list
        if not isinstance(idx_slots, list):
            idx_slots = [idx_slots]

        ## Check if wait_for_load is a list
        if not isinstance(wait_for_load, list):
            wait_for_load = [wait_for_load] * len(idx_slots)

        print(f"FR: Loading slots {idx_slots} in the background. Waiting: {wait_for_load}") if self._verbose > 1 else None
        print(f"FR: Loaded: {self.loaded}, Loading: {self.loading}") if self._verbose > 1 else None
        thread = None
        for idx_slot, wait in zip(idx_slots, wait_for_load):
            ## Check if slot is already loaded
            (print(f"FR: Slot {idx_slot} already loaded") if (idx_slot in self.loaded) else None) if self._verbose > 1 else None
            (print(f"FR: Slot {idx_slot} already loading") if (idx_slot in self.loading) else None) if self._verbose > 1 else None
            ## If the slot is not already loaded or loading
            if (idx_slot not in self.loading) and (idx_slot not in self.loaded):
                print(f"FR: Loading slot {idx_slot}") if self._verbose > 1 else None
                ## Load the slot
                self.loading.append(idx_slot)
                thread = threading.Thread(target=self._load_slot, args=(idx_slot, thread))
                thread.start()

                ## Wait for the slot to load if wait_for_load is True
                if wait:
                    print(f"FR: Waiting for slot {idx_slot} to load") if self._verbose > 1 else None
                    thread.join()
                    print(f"FR: Slot {idx_slot} loaded") if self._verbose > 1 else None
            ## If the slot is already loading
            elif idx_slot in self.loading:
                ## Wait for the slot to load if wait_for_load is True
                if wait:
                    print(f"FR: Waiting for slot {idx_slot} to load") if self._verbose > 1 else None
                    while idx_slot in self.loading:
                        time.sleep(0.01)
                    print(f"FR: Slot {idx_slot} loaded") if self._verbose > 1 else None

    def _load_slot(self, idx_slot: tuple, blocking_thread: threading.Thread=None):
        """
        Load a single slot.
        self.slots[idx_slot[0]][idx_slot[1]] will be populated
         with the loaded data.
        Allows for a blocking_thread argument to be passed in,
         which will force this new thread to wait until the
         blocking_thread is finished (join()) before loading.
        
        Args:
            idx_slot (tuple):
                Tuple containing the indices of the slot to load.
                Should be of the form (idx_video, idx_buffer).
            blocking_thread (threading.Thread):
                Thread to wait for before loading.
        """
        ## Set backend of decord to PyTorch
        decord.bridge.set_bridge(self._decord_backend)
        ## Wait for the previous slot to finish loading
        if blocking_thread is not None:
            blocking_thread.join()
        ## Load the slot
        idx_video, idx_buffer = idx_slot
        idx_frame_start, idx_frame_end = self.boundaries[idx_video][idx_buffer]
        loaded = False
        while loaded == False:
            try:
                self.slots[idx_video][idx_buffer] = self.video_readers[idx_video][idx_frame_start:idx_frame_end+1]
                loaded = True
                print(f"FR: Loaded slot {idx_slot} with frames {idx_frame_start} to {idx_frame_end} from video {idx_video}") if self._verbose > 1 else None
            except Exception as e:
                print(f"FR WARNING: Failed to load slot {idx_slot}. Likely causes are: 1) File is partially corrupted, 2) You are trying to go back to a file that was recently removed from a slot.") if self._verbose > 0 else None
                print(f"    Sleeping for 1s, then will try loading again. Decord error below:") if self._verbose > 0 else None
                print(e)
                time.sleep(1)

        ## Mark the slot as loaded
        self.loaded.append(idx_slot)
        ## Remove the slot from the loading list
        self.loading.remove(idx_slot)
                
    def _delete_slots(self, idx_slots: list):
        """
        Delete slots from memory.
        Sets self.slots[idx_slot[0]][idx_slot[1]] to None.

        Args:
            idx_slots (list):
                List of tuples containing the indices of the 
                 slots to delete.
                Each tuple should be of the form (idx_video, idx_buffer).
        """
        print(f"FR: Deleting slots {idx_slots}") if self._verbose > 1 else None
        ## Find all loaded slots
        idx_loaded = [idx_slot for idx_slot in idx_slots if idx_slot in self.loaded]
        for idx_slot in idx_loaded:
            ## If the slot is loaded
            if idx_slot in self.loaded:
                ## Delete the slot
                self.slots[idx_slot[0]][idx_slot[1]] = None
                ## Remove the slot from the loaded list
                self.loaded.remove(idx_slot)
                print(f"FR: Deleted slot {idx_slot}") if self._verbose > 1 else None

    def delete_all_slots(self):
        """
        Delete all slots from memory.
        Uses the _delete_slots() method.
        """
        print(f"FR: Deleting all slots") if self._verbose > 1 else None
        self._delete_slots(self.loaded)

    def wait_for_loading(self):
        """
        Wait for all slots to finish loading.
        """
        print(f"FR: Waiting for all slots to load") if self._verbose > 1 else None
        while len(self.loading) > 0:
            time.sleep(0.01)
        

    
    def get_frames_from_single_video_index(self, idx: tuple):
        """
        Get a slice of frames by specifying the video number and 
         the frame number.

        Args:
            idx (tuple or int):
            A tuple containing the index of the video and a slice for the frames.
            (idx_video: int, idx_frames: slice)
            If idx is an int or slice, it is assumed to be the index of the video, and
             a new BufferedVideoReader(s) will be created with just those videos.

        Returns:
            frames (torch.Tensor):
                A tensor of shape (num_frames, height, width, num_channels)
        """
        ## if idx is an int or slice, use idx to make a new BufferedVideoReader of just those videos
        idx = slice(idx, idx+1) if isinstance(idx, int) else idx
        if isinstance(idx, slice):
            ## convert to a slice
            print(f"FR: Returning new buffered video reader(s). Videos={idx.start} to {idx.stop}.") if self._verbose > 1 else None
            return BufferedVideoReader(
                video_readers=self.video_readers[idx],
                buffer_size=self.buffer_size,
                prefetch=self.prefetch,
                method_getitem='continuous',
                starting_seek_position=0,
                decord_backend='torch',
                decord_ctx=None,
                verbose=self._verbose,
            )
        print(f"FR: Getting item {idx}") if self._verbose > 1 else None
        ## Assert that idx is a tuple of (int, int) or (int, slice)
        assert isinstance(idx, tuple), f"idx must be: int, tuple of (int, int), or (int, slice). Got {type(idx)}"
        assert len(idx) == 2, f"idx must be: int, tuple of (int, int), or (int, slice). Got {len(idx)} elements"
        assert isinstance(idx[0], int), f"idx[0] must be an int. Got {type(idx[0])}"
        assert isinstance(idx[1], int) or isinstance(idx[1], slice), f"idx[1] must be an int or a slice. Got {type(idx[1])}"
        ## Get the index of the video and the slice of frames
        idx_video, idx_frames = idx
        ## If idx_frames is a single integer, convert it to a slice
        idx_frames = slice(idx_frames, idx_frames+1) if isinstance(idx_frames, int) else idx_frames
        ## Bound the range of the slice
        idx_frames = slice(max(idx_frames.start, 0), min(idx_frames.stop, len(self.video_readers[idx_video])))
        ## Assert that slice is not empty
        assert idx_frames.start < idx_frames.stop, f"Slice is empty: idx:{idx}"

        ## Get the start and end indices for the slice of frames
        idx_frame_start = idx_frames.start if idx_frames.start is not None else 0
        idx_frame_end = idx_frames.stop if idx_frames.stop is not None else len(self.video_readers[idx_video])
        idx_frame_step = idx_frames.step if idx_frames.step is not None else 1

        ## Get the indices of the slots that contain the frames
        idx_slots = [(idx_video, i) for i in range(idx_frame_start // self.buffer_size, ((idx_frame_end-1) // self.buffer_size)+1)]
        print(f"FR: Slots to load: {idx_slots}") if self._verbose > 1 else None

        ## Load the prefetch slots
        idx_slot_lookuptable = np.where((self.lookup['video']==idx_slots[-1][0]) * (self.lookup['slot']==idx_slots[-1][1]))[0][0]
        if self.prefetch > 0:
            idx_slots_prefetch = [(self.lookup['video'][ii], self.lookup['slot'][ii]) for ii in range(idx_slot_lookuptable+1, idx_slot_lookuptable+self.prefetch+1) if ii < len(self.lookup)]
        else:
            idx_slots_prefetch = []
        ## Load the slots
        self._load_slots(idx_slots + idx_slots_prefetch, wait_for_load=[True]*len(idx_slots) + [False]*len(idx_slots_prefetch))

        ## Get the frames from the slots
        idx_frames_slots = [slice(max(idx_frame_start - self.boundaries[idx_slot[0]][idx_slot[1]][0], 0), min(idx_frame_end - self.boundaries[idx_slot[0]][idx_slot[1]][0], self.buffer_size), idx_frame_step) for idx_slot in idx_slots]
        print(f"FR: Frames within slots: {idx_frames_slots}") if self._verbose > 1 else None

        ## Get the frames. Then concatenate them along the first dimension using torch.cat
        ### Skip the concatenation if there is only one slot
        if len(idx_slots) == 1:
            frames = self.slots[idx_slots[0][0]][idx_slots[0][1]][idx_frames_slots[0]]
        else:
            print(f"FR: Warning. Slicing across multiple slots is SLOW. Consider increasing buffer size or adjusting batching method.") if self._verbose > 1 else None
            frames = torch.cat([self.slots[idx_slot[0]][idx_slot[1]][idx_frames_slot] for idx_slot, idx_frames_slot in zip(idx_slots, idx_frames_slots)], dim=0)
        
        ## Delete the slots that are no longer needed. 
        ### Find slots before the posthold to delete
        idx_slots_delete = [(self.lookup['video'][ii], self.lookup['slot'][ii]) for ii in range(idx_slot_lookuptable-self.posthold) if ii >= 0]
        ### Delete all previous slots
        self._delete_slots(idx_slots_delete)
        # ### All slots from old videos should be deleted.
        # self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] < idx_video])
        # ### All slots from previous buffers should be deleted.
        # self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] == idx_video and idx_slot[1] < idx_frame_start // self.buffer_size])

        # ## Squeeze if there is only one frame
        # frames = frames.squeeze(0) if frames.shape[0] == 1 else frames

        return frames

    def get_frames_from_continuous_index(self, idx):
        """
        Get a batch of frames from a continuous index.
        Here the videos are treated as one long sequence of frames,
         and the index is the index of the frames in this sequence.

        Args:
            idx (int or slice):
                The index of the frames to get. If an int, a single frame is returned.
                If a slice, a batch of frames is returned.

        Returns:
            frames (torch.Tensor):
                A tensor of shape (num_frames, height, width, num_channels)
        """
        ## Assert that idx is an int or a slice
        assert isinstance(idx, (int, np.int_)) or isinstance(idx, slice), f"idx must be an int or a slice. Got {type(idx)}"
        idx = int(idx) if isinstance(idx, (np.int_)) else idx
        ## If idx is a single integer, convert it to a slice
        idx = slice(idx, idx+1) if isinstance(idx, int) else idx
        ## Assert that the slice is not empty
        assert idx.start < idx.stop, f"Slice is empty: idx:{idx}"
        ## Assert that the slice is not out of bounds
        assert idx.stop <= self.num_frames_total, f"Slice is out of bounds: idx:{idx}"
        
        ## Find the video and frame indices
        idx_video_start = np.searchsorted(self._cumulative_frame_start, idx.start, side='right') - 1
        idx_video_end = np.searchsorted(self._cumulative_frame_end, idx.stop, side='left')
        ## Get the frames using the __getitem__ method
        ### This needs to be done one video at a time
        frames = []
        for idx_video in range(idx_video_start, idx_video_end+1):
            ## Get the start and end indices for the slice of frames
            idx_frame_start = idx.start - self._cumulative_frame_start[idx_video] if idx_video == idx_video_start else 0
            idx_frame_end = idx.stop - self._cumulative_frame_start[idx_video] if idx_video == idx_video_end else len(self.video_readers[idx_video])
            ## Get the frames
            print(f"FR: Getting frames from video {idx_video} from {idx_frame_start} to {idx_frame_end}") if self._verbose > 1 else None
            frames.append(self.get_frames_from_single_video_index((idx_video, slice(idx_frame_start, idx_frame_end, idx.step))))
        ## Concatenate the frames if there are multiple videos
        frames = torch.cat(frames, dim=0) if len(frames) > 1 else frames[0]

        return frames

    def set_iterator_frame_idx(self, idx):
        """
        Set the starting frame for the iterator.
        Index should be in 'continuous' format.

        Args:
            idx (int):
                The index of the frame to start the iterator from.
                Should be in 'continuous' format where the index
                 is the index of the frame in the entire sequence 
                 of frames.
        """
        self._iterator_frame = idx
        
    def __getitem__(self, idx):
        if self.method_getitem == 'by_video':
            ## handle None values in the slice idx
            if isinstance(idx, slice):
                idx = slice(
                    idx.start if idx.start is not None else 0,
                    idx.stop if idx.stop is not None else self.num_videos,
                    idx.step if idx.step is not None else 1,
                )
            return self.get_frames_from_single_video_index(idx)
        elif self.method_getitem == 'continuous':
            ## handle None values in the slice idx
            if isinstance(idx, slice):
                idx = slice(
                    idx.start if idx.start is not None else 0,
                    idx.stop if idx.stop is not None else self.num_frames_total,
                    idx.step if idx.step is not None else 1,
                )
            return self.get_frames_from_continuous_index(idx)
        else:
            raise ValueError(f"Invalid method_getitem: {self.method_getitem}")

    def __len__(self): 
        if self.method_getitem == 'by_video':
            return len(self.video_readers)
        elif self.method_getitem == 'continuous':
            return self.num_frames_total
    def __repr__(self): 
        if self.method_getitem == 'by_video':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, method_getitem='{self.method_getitem}', loaded={self.loaded}, prefetch={self.prefetch}, loading={self.loading}, verbose={self._verbose})"    
        elif self.method_getitem == 'continuous':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, total_frames={self.num_frames_total}, method_getitem='{self.method_getitem}', iterator_frame={self._iterator_frame}, prefetch={self.prefetch}, loaded={self.loaded}, loading={self.loading}, verbose={self._verbose})"
    def __iter__(self): 
        """
        If method_getitem is 'by_video':
            Iterate over BufferedVideoReaders for each video.
        If method_getitem is 'continuous':
            Iterate over the frames in the video.
            Makes a generator that yields single frames directly from
            the buffer slots.
            If it is the initial frame, or the first frame of a slot,
            then self.get_frames_from_continuous_index is called to
            load the next slots into the buffer.
        """
        if self.method_getitem == 'by_video':
            return iter([BufferedVideoReader(
                video_readers=[self.video_readers[idx]],
                buffer_size=self.buffer_size,
                prefetch=self.prefetch,
                method_getitem='continuous',
                starting_seek_position=0,
                decord_backend='torch',
                decord_ctx=None,
                verbose=self._verbose,
            ) for idx in range(len(self.video_readers))])
        elif self.method_getitem == 'continuous':
            ## Initialise the buffers by loading the first frame in the sequence
            self.get_frames_from_continuous_index(self._iterator_frame)
            ## Make lazy iterator over all frames
            def lazy_iterator():
                while self._iterator_frame < self.num_frames_total:
                    ## Find slot for current frame idx
                    idx_video = np.searchsorted(self._cumulative_frame_start, self._iterator_frame, side='right') - 1
                    idx_slot_in_video = (self._iterator_frame - self._cumulative_frame_start[idx_video]) // self.buffer_size
                    idx_frame = self._iterator_frame - self._cumulative_frame_start[idx_video]
                    ## If the frame is at the beginning of a slot, then use get_frames_from_single_video_index otherwise just grab directly from the slot
                    if (self._iterator_frame in self._start_frame_continuous):
                        yield self.get_frames_from_continuous_index(self._iterator_frame)[0]
                    else:
                    ## Get the frame directly from the slot
                        yield self.slots[idx_video][idx_slot_in_video][idx_frame%self.buffer_size]
                    self._iterator_frame += 1
        return iter(lazy_iterator())


# def make_tiled_video_array_old(
#     paths_videos, 
#     frame_idx_list, 
#     block_height_width=[300,300],
#     n_channels=3, 
#     tiling_shape=None, 
#     dtype=np.uint8,
#     interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
#     crop_idx=None,
#     overlay_signals=None,
#     overlay_idx=None,
#     overlay_color=[0,0.7,1.0],
#     spacer_black_frames=0,
#     pixel_val_range=None,
#     verbose=True,
# ):
#     """
#     Creates a tiled video array from a list of paths to videos.
#     NOTE: On my Ubuntu machine:
#         - importing 'av' after cv2 causes cv2.imshow to hang forever if
#          cv2.imshow is not called first
#         - importing 'decord' after cv2 causes cv2.imshow to crash the kernel
#          if cv2.imshow is not called first
#     RH 2022

#     Args:
#         paths_videos (list of str):
#             List of paths to videos.
#         frame_idx_list (ndarray, 3D, int):
#             Shape: (n_chunks, 2, n_videos)
#             Second dimension: (start_frame, end_frame) of chunk.
#             Values should be positive integers.
#             To insert black frames instead of video chunk, use
#              [-1, -1] for the idx tuple.
#         block_height_width (list or 2-tuple):
#             2-tuple or list of height and width of each block.
#         n_channels (int):
#             Number of channels.
#         tiling_shape (2-tuple):
#             2-tuple or list of height and width of the tiled video.
#             If None, then set to be square and large enough to 
#              contain all the blocks.
#         dtype (np.dtype):
#             Data type of the output array. Should match the data
#              type of the input videos.
#         interpolation (torchvision.transforms.InterpolationMode):
#             Interpolation mode for the video. Should be one of the
#              torchvision.transforms.InterpolationMode values.
#         crop_idx (list of 4-tuples):
#             List of 4-tuples or lists of indices to crop the video.
#             [top, bottom, left, right]
#             If None, then no cropping is performed.
#             Outer list should be same length as paths_videos. Each 
#              entry should correspond to the crop indices for a video.
#         overlay_signals (list of ndarray):
#             List of signals to overlay on the video.
#             Each signal should be a numpy array of shape(frames, n_channels).
#             The signals will be represented as a white rectangle with
#              indices from overlay_idx.
#         overlay_idx (list or 4-tuple):
#             List of indices to overlay the signals.
#             [top, bottom, left, right]
#         overlay_color (list or 3-tuple of floats):
#             Color of the overlay.
#             range: 0 to 1
#         spacer_black_frames:
#             Number of black frames to add between each chunk.
#         pixel_val_range:
#             2-tuple or list of the minimum and maximum pixel values.
#             If None, then no clipping is performed.
#         verbose:
#             Whether to print progress.

#     Returns:
#         output video array:
#             Tiled video array.
#             shape(frames, tiling_shape[0]*block_height_width[0], tiling_shape[1]*block_height_width[1], channels)
#     """
#     import av
#     import decord

#     ##  Example values
#     ##  - frame_idx_list = [np.array([[703,843], [743,883], [799, 939], [744, 884]]*2).T, np.array([[39,89], [43,93], [99, 149], [44, 94]]*2).T]
#     ##  - frame_idx_list = [np.array([[37900,38050], [37900,38050], [37900,38050], [37900,38050]]*2).T, np.array([[37900,38050], [37900,38050], [37900,38050], [37900,38050]]*2).T]
    
#     ##  - roi = plotting_helpers.select_ROI(image)
#     ##  - pts = np.array(roi.selected_points).squeeze().astype('uint32')
#     ##  - crop_idx = [pts[0,1], pts[1,1], pts[0,0], pts[1,0]]*8
#     ##  - block_height_width=[crop_idx[1]-crop_idx[0] , crop_idx[3]-crop_idx[2]]
#     ##  - tiling_shape = [2,2]
#     ##  - spacer_black_frames = 5
#     ##  - paths_videos = [path_video] * 8


#     def resize_torch(images, new_shape=[100,100], interpolation=interpolation):
#         resize = torchvision.transforms.Resize(new_shape, interpolation=interpolation, max_size=None, antialias=None)
#         return resize(torch.as_tensor(images)).numpy()

#     def add_overlay(chunk, overlay_signal, overlay_idx, overlay_color=[1,1,1]):
#         ol_height = overlay_idx[1]-overlay_idx[0]
#         ol_width = overlay_idx[3]-overlay_idx[2]
#         ol = np.ones((chunk.shape[0], ol_height, ol_width, chunk.shape[3])) * overlay_signal[:,None,None,None] * np.array(overlay_color)[None,None,None,:]
#         chunk[:, overlay_idx[0]:overlay_idx[1], overlay_idx[2]:overlay_idx[3], :] = ol
#         return chunk

#     duration_chunks = frame_idx_list[:,1,:] - frame_idx_list[:,0,:] + spacer_black_frames
#     max_frames_per_chunk = np.nanmax(duration_chunks, axis=1)

#     null_chunks = (frame_idx_list == -1).all(axis=1)
    
#     ## ASSERTIONS
#     ## check to make sure that shapes are correct
#     for i_chunk, chunk in enumerate(frame_idx_list):
#         assert chunk.shape[0] == 2, f'RH ERROR: size of first dimension of each frame_idx matrix should be 2'
#         assert chunk.shape[1] == len(paths_videos), f'RH ERROR: size of second dimension of each frame_idx matrix should match len(paths_videos)'

#     n_vids = len(paths_videos)
#     n_frames_total = max_frames_per_chunk.sum()  ## total number of frames in the final video
#     block_aspect_ratio = block_height_width[0] / block_height_width[1]

#     cum_start_idx_chunk = np.cumsum(np.concatenate(([0], max_frames_per_chunk)))[:-1] ## cumulative starting indices of temporal chunks in final video

#     if tiling_shape is None:
#         el = int(np.ceil(np.sqrt(n_vids)))  ## 'edge length' in number of videos
#         tiling_shape = [el, el]  ## n_vids high , n_vids wide
#     tile_grid_tmp = np.meshgrid(np.arange(tiling_shape[0]), np.arange(tiling_shape[1]))
#     tile_position_vids = [np.reshape(val, -1, 'F') for val in tile_grid_tmp]  ## indices of tile/block positions for each video

#     vid_height_width = list(np.array(block_height_width) * tiling_shape)  ## total height and width of final video


#     tile_topLeft_idx = [[tile_position_vids[0][i_vid]*block_height_width[0], tile_position_vids[1][i_vid]*block_height_width[1]] for i_vid in range(len(paths_videos))]  ## indices of the top left pixels for each tile/block. List of lists: outer list is tile/block, inner list is [y,x] starting idx

#     video_out = np.zeros((n_frames_total, vid_height_width[0], vid_height_width[1], n_channels), dtype)  ## pre-allocation of final video array

#     vid_dict = {}  ## pre-allocation of video dictionary

#     ## Set decord backend to PyTorch
#     decord.bridge.set_bridge('torch')

#     for i_vid, path_vid in enumerate(tqdm(paths_videos, leave=False)):
#         if isinstance(path_vid, list):
#             flag_multivid = True
#             multivid_lens = [av.open(str(path)).streams.video[0].frames for path in path_vid]
#             # multivid_lens = [len(decord.VideoReader(path)) for path in path_vid]  ## decord method of same thing
#             cum_start_idx_multiVid = np.cumsum(np.concatenate(([0], multivid_lens)))[:-1]
#         else:
#             flag_multivid = False

#         for i_chunk, idx_chunk in enumerate(tqdm(frame_idx_list, leave=False)):
#             if null_chunks[i_chunk, i_vid]:
#                 continue
#             elif flag_multivid:
#                 frames_remainder = idx_chunk[1,i_vid] - idx_chunk[0,i_vid]  ## initialization of remaining frames
#                 frame_toStartGlobal = idx_chunk[0,i_vid]  ## frame to start at (in concatenated frame indices)

#                 chunks_list = []
#                 while frames_remainder > 0:
#                     multivid_toStart = indexing.get_last_True_idx((frame_toStartGlobal - cum_start_idx_multiVid) >= 0)  ## which of the multivids to start at

#                     frame_toStartInVid = frame_toStartGlobal - cum_start_idx_multiVid[multivid_toStart]  ## where to start in the vid

#                     frames_toEndOfVid = multivid_lens[multivid_toStart] - frame_toStartInVid  ## number of frames left in the vid
#                     frames_toGrab = min(frames_remainder  ,  frames_toEndOfVid)  ## number of frames to get from current vid
#                     frames_remainder -= frames_toGrab

#                     if path_vid[multivid_toStart] in vid_dict.keys():
#                         print(f'using cached video {path_vid[multivid_toStart]}') if verbose else None
#                     else:
#                         vid_dict[path_vid[multivid_toStart]] = decord.VideoReader(str(path_vid[multivid_toStart]), ctx=decord.cpu())  ## open the vid
#                         print(f'opening new video file {path_vid[multivid_toStart]}') if verbose else None
#                     vid = vid_dict[path_vid[multivid_toStart]]
                    
#                     chunks_list.append(vid[frame_toStartInVid : frame_toStartInVid+frames_toGrab].numpy())  ## raw video chunk
#                     frame_toStartGlobal += frames_toGrab

#                 chunk = np.concatenate(chunks_list, axis=0)
#             else:
#                 vid = decord.VideoReader(path_vid, ctx=decord.cpu())
#                 chunk = vid[idx_chunk[0, i_vid] : idx_chunk[1, i_vid]].numpy()  ## raw video chunk

#             chunk_height, chunk_width, chunk_n_frames, chunk_n_channels = chunk.shape
#             if crop_idx is not None:
#                 chunk = chunk[:, crop_idx[i_vid][0]:crop_idx[i_vid][1], crop_idx[i_vid][2]:crop_idx[i_vid][3], :]

#             ## first we get the aspect ratio right by padding to correct aspect ratio
#             aspect_ratio = chunk.shape[1] / chunk.shape[2]
#             if aspect_ratio >= block_aspect_ratio:
#                 tmp_height = chunk.shape[1]
#                 tmp_width = int(np.ceil(chunk.shape[1] / block_aspect_ratio))
#             if aspect_ratio < block_aspect_ratio:
#                 tmp_height = int(np.ceil(chunk.shape[2] * block_aspect_ratio))
#                 tmp_width = chunk.shape[2]
#             chunk_ar = image_processing.center_pad_images(chunk, height_width=[tmp_height, tmp_width])

#             ## then we resize the movie to the final correct size
#             chunk_rs = resize_torch(chunk_ar.transpose(0,3,1,2), new_shape=block_height_width, interpolation=interpolation).transpose(0,2,3,1)

#             if pixel_val_range is not None:
#                 chunk_rs[chunk_rs < pixel_val_range[0]] = pixel_val_range[0]  ## clean up interpolation errors
#                 chunk_rs[chunk_rs > pixel_val_range[1]] = pixel_val_range[1]

#             ## add overlay to the chunk
#             if overlay_signals is not None:
#                 add_overlay(
#                     chunk_rs, 
#                     overlay_signals[i_vid][idx_chunk[0,i_vid]:idx_chunk[1,i_vid], i_chunk], 
#                     overlay_idx,
#                     overlay_color=overlay_color,
#                     )


#             ## drop into final video array
#             video_out[
#                 cum_start_idx_chunk[i_chunk] : duration_chunks[i_chunk, i_vid] + cum_start_idx_chunk[i_chunk] - spacer_black_frames,
#                 tile_topLeft_idx[i_vid][0] : tile_topLeft_idx[i_vid][0]+block_height_width[0], 
#                 tile_topLeft_idx[i_vid][1] : tile_topLeft_idx[i_vid][1]+block_height_width[1], 
#                 :
#             ] = chunk_rs

#     return video_out


def make_tiled_video_array(
    videos: List[np.ndarray],
    shape: Optional[Tuple[int, int]]=None,
    verbose: bool=True,
):
    """
    Make a video array from a list of ndarrays of videos.
    Order of videos is top to bottom, then left to right.
    RH 2024

    Args:
        videos (List[np.ndarray]):
            List of video arrays.
            Shape: (frames, height, width, channels) or (frames, height, width)
            All videos should have the same dtype.
        shape (Tuple[int, int]):
            Shape of the final video array.
            (height, width)
            If None, then the videos will be tiled into the smallest square.
            Videos are indexed from top to bottom, then left to right.
        verbose (bool):
            Whether to print progress.
    """
    ## Check inputs
    assert isinstance(videos, list), f"videos must be a list. Got {type(videos)}"
    assert all([isinstance(video, np.ndarray) for video in videos]), f"All elements of videos must be numpy arrays. Got {[type(video) for video in videos]}"
    assert all([video.dtype == videos[0].dtype for video in videos]), f"All videos must have the same dtype. Got {[video.dtype for video in videos]}"
    if shape is not None:
        assert isinstance(shape, tuple), f"shape must be a tuple. Got {type(shape)}"
        assert len(shape) == 2, f"shape must be a 2-tuple. Got {len(shape)} elements"
        assert all([isinstance(val, int) for val in shape]), f"shape must contain only integers. Got {shape}"
    
    n_videos = len(videos)
    
    ## Set the output array shape
    if shape is None:
        n_height = int(np.floor(np.sqrt(n_videos)))
        n_width = int(np.ceil(np.sqrt(n_videos)))
        shape = (n_height, n_width)
    assert shape[0] * shape[1] >= n_videos, f"shape[0] * shape[1] must be greater than or equal to the number of videos. Got {shape[0] * shape[1]} < {n_videos}"
    print(f"Making video array with shape: {shape} videos") if verbose else None

    ## Pad the videos' dimensions to 4D
    for ii, video in enumerate(videos):
        assert video.ndim in [3,4], f"videos[{ii}] must be 3D or 4D. Got {video.ndim} dimensions"
        if video.ndim == 3:
            videos[ii] = video[..., None]

    ## Get the shapes of each video
    n_frames, heights, widths, channels = (np.array([video.shape[ii] for video in videos]) for ii in range(4))
    max_frames = max(n_frames)
    print(f"Max video length and total frames in output video array: {max_frames}") if verbose else None

    ## Get the dtype of the videos
    dtype = videos[0].dtype  ## all videos should have the same dtype given the assertions above
    print(f"Video dtype: {dtype}") if verbose else None

    ## Assign video array indices for each video
    idx_videoArray_videos = np.array([(ii % shape[0], ii // shape[0]) for ii in range(n_videos)])
    print(f"Video array indices: {idx_videoArray_videos}") if verbose else None
    ## Get total lengths and heights of the final array
    ### Get max width for each column
    widths_col = np.array([np.max(widths[idx_videoArray_videos[:,1] == ii]) for ii in np.unique(idx_videoArray_videos[:,1])])
    ### Get max height for each row
    heights_row = np.array([np.max(heights[idx_videoArray_videos[:,0] == ii]) for ii in np.unique(idx_videoArray_videos[:,0])])
    ### Make final shape
    final_shape = (max_frames, np.sum(heights_row), np.sum(widths_col), channels[0])
    video_array = np.zeros(final_shape, dtype)
    print(f"Final output array shape: {video_array.shape}") if verbose else None

    ## Fill the video array
    ### Get idx_top for each video
    idx_tops_rows = np.cumsum(np.concatenate(([0], heights_row)))[:-1]
    idx_tops_cols = np.cumsum(np.concatenate(([0], widths_col)))[:-1]
    print(f"idx_tops_rows: {idx_tops_rows}, idx_tops_cols: {idx_tops_cols}") if verbose else None
    for ii, video in enumerate(videos):
        idx_top = idx_tops_rows[idx_videoArray_videos[ii,0]]
        idx_left = idx_tops_cols[idx_videoArray_videos[ii,1]]
        ### Fill the video array
        video_array[:video.shape[0], idx_top:idx_top+video.shape[1], idx_left:idx_left+video.shape[2], :] = video

    return video_array


def add_signal_overlay(
    vid: np.ndarray,
    signal: np.ndarray,
    normalize: bool=True,
    width: int=10,
    overlay_or_concatenate: str='overlay',
    position: str='bottom',
    color: Tuple[float, float, float]=(1,1,1),
) -> np.ndarray:
    """
    Add a signal overlay to a video array.
    RH 2024

    Args:
        vid (np.ndarray):
            Video array to add overlay to.
            Shape: (frames, height, width, channels) or (frames, height, width)
            len(frames) must be equal to vid.shape[0].
        signal (np.ndarray):
            Signal to overlay.
            Shape: (frames, n_channels) or (frames,)
            len(frames) must be equal to vid.shape[0].
            Either signal.ndim must be 1 OR n_channels must be equal to
            vid.shape[-1].
        normalize (bool):
            Whether to normalize the signal. \n
            If True: \n
                If vid.dtype is a float type, then the max value of the signal
                is normalized to 1. If vid.dtype is an integer type, then the
                max value of the signal is normalized to the maximum value of
                the integer type. \n
            If False: \n
                The signal and the vid must have the same dtype.
        width (int):
            Width of the overlay in pixels.  
        overlay_or_concatenate (str):
            Whether to 'overlay' the signal on top of the video or to 'concatenate' the signal to the video.
            Options: 'overlay', 'concatenate'
        position (str):
            Position to overlay the signal.
            Options: 'top', 'bottom'
        color (Union[Tuple[float, float, float], List[float], np.ndarray):
            Color of the overlay.
            Shape: (n_channels,)
            range: 0 to 1
    """
    ## Check inputs
    assert isinstance(vid, np.ndarray), f"vid must be a numpy array. Got {type(vid)}"
    assert isinstance(signal, np.ndarray), f"signal must be a numpy array. Got {type(signal)}"
    assert vid.ndim in [3,4], f"vid must be 3D or 4D. Got {vid.ndim} dimensions"
    vid = vid[..., None] if vid.ndim == 3 else vid
    assert signal.ndim in [1,2], f"signal must be 1D or 2D. Got {signal.ndim} dimensions"
    signal = signal[..., None] if signal.ndim == 1 else signal
    assert vid.shape[0] == signal.shape[0], f"len(frames) of vid and signal must be equal. Got {vid.shape[0]} != {signal.shape[0]}"
    assert (signal.shape[1] == 1) or (signal.shape[1] == vid.shape[-1]), f"signal must have 1 channel or the same number of channels as vid. Got {signal.shape[1]} channels in signal and {vid.shape[-1]} channels in vid"
    if not normalize:
        assert vid.dtype == signal.dtype, f"vid and signal must have the same dtype if normalize is False. Got {vid.dtype} and {signal.dtype}"
    assert isinstance(width, int), f"width must be an integer. Got {type(width)}"
    assert isinstance(color, (tuple, list, np.ndarray)), f"color must be a tuple, list, or numpy array. Got {type(color)}"
    
    ## Normalize the signal
    if normalize:
        if np.issubdtype(vid.dtype, np.floating):
            signal = (signal / np.max(signal)).astype(vid.dtype)
        else:
            signal = ((signal / np.max(signal)) * np.iinfo(vid.dtype).max).astype(vid.dtype)
            
    ## Get the color
    color = np.array(color, dtype=vid.dtype)

    ## Make the overlay
    if position in ['top', 'bottom']:
        overlay = np.ones((vid.shape[0], width, vid.shape[2], vid.shape[3],), dtype=vid.dtype)
    elif position in ['left', 'right']:
        overlay = np.ones((vid.shape[0], vid.shape[1], width, vid.shape[3],), dtype=vid.dtype)
    else:
        raise ValueError(f"Invalid position: {position}")
    overlay = overlay * color[None,None,None,:] * signal[:,None,None,:]
    
    ## Overlay the signal
    if overlay_or_concatenate == 'overlay':
        if position == 'top':
            vid[:, :width, :, :] = overlay
        elif position == 'bottom':
            vid[:, -width:, :, :] = overlay
        elif position == 'left':
            vid[:, :, :width, :] = overlay
        elif position == 'right':
            vid[:, :, -width:, :] = overlay
    elif overlay_or_concatenate == 'concatenate':
        if position == 'top':
            vid = np.concatenate([overlay, vid], axis=1)
        elif position == 'bottom':
            vid = np.concatenate([vid, overlay], axis=1)
        elif position == 'left':
            vid = np.concatenate([overlay, vid], axis=2)
        elif position == 'right':
            vid = np.concatenate([vid, overlay], axis=2)
    else:
        raise ValueError(f"Invalid overlay_or_concatenate: {overlay_or_concatenate}")
    
    return vid


def save_array_as_video_ffmpeg(
    array,
    filepath,
    frame_rate=30,
    quality=25,
    lossless=False,
    codec='libx264',
    pixel_format='yuv420p',
    speed='medium',
    overwrite=False,
    verbose=True,
    **kwargs,
):
    """
    Saves a video array to a video file using ffmpegio. Defaults to using x264
    codec with yuv420p pixel format. This is a function that just scratches the
    surface of what ffmpeg can do. For more advanced options, check ffmpeg docs and pass
    them in as kwargs.
    RH 2024

    Args:
        array (ndarray):
            Video array to save. \n
            Dtype: uint8 for pixel_format='yuv420p', check ffmpegio and ffmpeg
            docs otherwise. \n
            Shape: (frames, height, width, channels) or (frames, height, width)
            or (frames, height, width, channels, 1) \n
        filepath (str):
            Path to save video file to.
        frame_rate (float):
            Frame rate of the video.
        quality (int):
            Quality of the video. (0 to 100)
            This is converted to 'crf' for 'libx264' codec.
        lossless (bool):
            Whether or not to use compression.
        codec (str):
            Codec to use. 'libx264' is a good default. \n
            Common codecs: \n
                * 'libx264' (H.264): Decent quality, fast, oftend doesn't work with decord \n
                * 'png' (PNG): Large, lossless, slow, works with decord and most players \n
                * 'libx265' (H.265) \n
                * 'libvpx-vp9' (VP9) \n
        pixel_format (str):
            Pixel format to use. 'yuv420p' is a good default for 'libx264'
            codec.
        speed (str):
            Speed of compression. For 'libx264', you can use the following: \n
                * ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo \n
        overwrite (bool):
            Whether to overwrite existing file.
        verbose (bool):
            Whether to print progress.
        **kwargs:
            Additional arguments to pass to ffmpegio.video.write(). You can pass
            ffmpeg options as keyword arguments.
    """
    import ffmpegio

    ## Check that array is a numpy array
    assert isinstance(array, np.ndarray), f"array must be a numpy array. Got {type(array)}"
    ## Check that array is either 3D or 4D
    assert array.ndim in [3,4], f"array must be 3D or 4D. Got {array.ndim} dimensions"

    ## Set the crf (quality) for 'libx264' codec
    if codec == 'libx264':
        q_max = 51
        crf = int(np.round((100 - np.clip(quality, 0, 100)) * q_max / 100))


    if Path(filepath).exists() and not overwrite:
        raise FileExistsError(f"File already exists: {filepath}")
        
    if verbose:
        print(f"Saving video to: {filepath}")

    # Options for ffmpeg
    ffmpeg_options = {
        'vcodec': codec,
        'pix_fmt': pixel_format,
        'an': None,
    }
    if (not lossless) and (codec == 'libx264'):
        ffmpeg_options.update({'crf': crf})
        ffmpeg_options.update({'preset': speed})
    ffmpeg_options.update(kwargs)

    # Writing the video file
    ffmpegio.video.write(
        url=filepath,
        rate_in=frame_rate,
        data=array,
        overwrite=overwrite,
        show_log=verbose,
        **ffmpeg_options,
    )

    if verbose:
        print("Video saved successfully.")


def save_array_as_video_imageio(
    array: Union[np.ndarray, List], 
    filepath: str, 
    frame_rate: int = 30,
    codec: str = 'libx264',
    quality: int = 9,
    pixel_format: str = 'yuv420p',
    speed: str = 'medium',
    overwrite: bool = False
) -> None:
    """
    Saves a 3D or 4D array (frames, height, width, [channels]) as an .avi file
    using imageio.

    Args:
        array (Union[np.ndarray, list]): 
            The 3D or 4D array of frames to save. The array can be grayscale
            (3D) or RGB (4D) with shape (frames, height, width) or (frames,
            height, width, channels).
        filepath (str): 
            The path to save the video file.
        frame_rate (int):
            The frame rate of the video. Defaults to 30.
        codec (str):
            Codec to use. 'libx264' is a good default. 
            Common codecs:
                * 'libx264' (H.264): Decent quality, fast, often doesn't work with decord
                * 'png' (PNG): Large, lossless, slow, works with decord and most players
                * 'libx265' (H.265)
                * 'libvpx-vp9' (VP9)
        quality (int):
            Quality of the video. (1 to 10 inclusive)
        pixel_format (str):
            Pixel format to use. 'yuv420p' is a good default for 'libx264' codec. Defaults to 'yuv420p'.
        speed (str):
            Speed of compression. For 'libx264', you can use the following:
                * ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
            Defaults to 'medium'.
        overwrite (bool):
            Whether to overwrite the existing file. Defaults to False.
    """
    import os
    import imageio
    
    # Ensure the array is a numpy array
    array = np.array(array)
    
    # Validate the shape of the array
    if array.ndim not in [3, 4]:
        raise ValueError("Array must be 3D (grayscale) or 4D (RGB).")
    
    # Check if file exists and handle overwrite
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"The file '{filepath}' already exists. Use overwrite=True to overwrite it.")
    
    # Determine if the input is grayscale (3D) or RGB (4D)
    is_grayscale = array.ndim == 3
    
    # Initialize the writer with additional parameters
    writer = imageio.get_writer(
        uri=filepath, 
        fps=frame_rate, 
        codec=codec, 
        quality=quality, 
        format='avi',
        ffmpeg_params=['-pix_fmt', pixel_format, '-preset', speed]
    )

    try:
        for frame in array:
            if is_grayscale:
                # Convert grayscale to RGB by stacking the same frame 3 times
                frame = np.stack([frame]*3, axis=-1)
            writer.append_data(frame)
    finally:
        writer.close()
