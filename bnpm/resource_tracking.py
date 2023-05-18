import datetime
from threading import Timer
from pathlib import Path

import psutil

class _Device_Checker_Base():
    """
    Superclass for checking resource utilization.
    Subclasses must have:
        - self.check_utilization() which returns info_changing dict
    """
    def __init__(self, verbose=1):
        """
        Initialize the class.

        Args:
            verbose (int):
                Verbosity level. 
                0: no print statements. 
                1: basic statements and warnings.
        """
        self._verbose = int(verbose)
                
    def log_utilization(self, path_save=None):
        """
        Logs current utilization info from device.
        If self.log does not exist, creates it, else appends to it.
        """
        info_changing = self.check_utilization()
        
        if not hasattr(self, 'log'):
            self.log = {}
            self._iter_log = 0

            ## Populate with keys
            for key in info_changing.keys():
                self.log[key] = {}
            print(f'Created self.log with keys: {self.log.keys()}') if self._verbose > 0 else None
        else:
            assert hasattr(self, '_iter_log'), 'self.log exists but self._iter_log does not'
            self._iter_log += 1

        ## Populate with values
        for key in info_changing.keys():
            self.log[key][self._iter_log] = info_changing[key]

        ## Save
        if path_save is not None:
            assert path_save.endswith('.csv'), 'path_save must be a .csv file'
            ## Check if file exists
            if not Path(path_save).exists():
                ## Make a .csv file with header
                with open(path_save, 'w') as f:
                    f.write(','.join(self.log.keys()) + '\n')
                ## Append to file
                with open(path_save, 'a') as f:
                    f.write(','.join([str(info_changing[key]) for key in self.log.keys()]) + '\n')
            ## Append to file
            else:
                with open(path_save, 'a') as f:
                    f.write(','.join([str(info_changing[key]) for key in self.log.keys()]) + '\n')

        return self.log
    
    
    def track_utilization(
        self, 
        interval=0.2,
        path_save=None,
    ):
        """
        Starts tracking utilization at specified interval and
         logs utilization to self.log using self.log_utilization().
        Creates a background thread (called self.fn_timer) that runs
         self.log_utilization() every interval seconds.

        Args:
            interval (float):
                Interval in seconds at which to log utilization.
                Minimum useful interval is 0.2 seconds.
            path_save (str):
                Path to save log to. If None, does not save.
                File should be a .csv file.
        """
        self.stop_tracking()
        ## Make a background thread that runs self.log_utilization() every interval seconds
        def log_utilization_thread():
            self.log_utilization(path_save=path_save)

        self.fn_timer = _RepeatTimer(interval, log_utilization_thread)
        self.fn_timer.start()
        
    def stop_tracking(self):
        """
        Stops tracking utilization by canceling self.fn_timer thread.
        """
        if hasattr(self, 'fn_timer'):
            self.fn_timer.cancel()

    def __del__(self):
        self.stop_tracking()


class NVIDIA_Device_Checker(_Device_Checker_Base):
    """
    Class for checking NVIDIA GPU utilization.
    Requires nvidia-ml-py3 package.
    """
    def __init__(self, device_index=None, verbose=1):
        """
        Initialize NVIDIA_Device_Checker class.
        Calls nvidia_smi.nvmlInit(), gets device handles, and gets static info.

        Args:
            device_index (int):
                Index of device to monitor. If None, will monitor device 0.
            verbose (int):
                Verbosity level. 
                0: no print statements. 
                1: basic statements and warnings.
        """
        try:
            import nvidia_smi
        except ImportError:
            raise ImportError('nvidia_smi package not found. Install with "pip install nvidia-ml-py3"')
        self.nvidia_smi = nvidia_smi
        super().__init__(verbose=verbose)
        
        ## Initialize
        nvidia_smi.nvmlInit()  ## This is needed to get device info

        ## Get device handles
        self._handles_allDevices = self.get_device_handles()
        n_device = len(self._handles_allDevices)
        if n_device == 1:
            self.handle = self._handles_allDevices[0]
            self.device_index = 0
            print(f'Found one device. Setting self.device_index to 0.') if self._verbose > 0 else None
        else:
            assert isinstance(device_index, int), 'Device index must be specified since multiple devices were found'
            assert device_index < n_device, f'Device index specified is greater tban the number of devices found: {n_device}'  
        
        ## Get static info
        self.info_static = {}
        self.info_static['device_name']  = nvidia_smi.nvmlDeviceGetName(self.handle)
        self.info_static['device_index'] = nvidia_smi.nvmlDeviceGetIndex(self.handle)
        self.info_static['memory_total'] = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle).total
        self.info_static['power_limit']  = nvidia_smi.nvmlDeviceGetPowerManagementLimit(self.handle)
    
    def get_device_handles(self):
        nvidia_smi = self.nvidia_smi
        return [nvidia_smi.nvmlDeviceGetHandleByIndex(i_device) for i_device in range(nvidia_smi.nvmlDeviceGetCount())]

    def check_utilization(self):
        """
        Retrieves current utilization info from device.
        Includes: current time, memory, power, and processor utilization, fan speed, and temperature.
        """
        nvidia_smi = self.nvidia_smi
        h = self.handle
        info_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(h)

        info_changing = {}
        
        info_changing['time'] = datetime.datetime.now()
        
        info_changing['memory_free'] = info_mem.free
        info_changing['memory_used'] = info_mem.used
        info_changing['memory_used_percentage'] = 100 * info_mem.used / info_mem.total

        info_changing['power_used'] = nvidia_smi.nvmlDeviceGetPowerUsage(h)
        info_changing['power_used_percentage'] = 100* info_changing['power_used'] / nvidia_smi.nvmlDeviceGetPowerManagementLimit(h)

        info_changing['processor_used_percentage'] = nvidia_smi.nvmlDeviceGetUtilizationRates(h).gpu

        info_changing['temperature'] = nvidia_smi.nvmlDeviceGetTemperature(h, nvidia_smi.NVML_TEMPERATURE_GPU)

        info_changing['fan_speed'] = nvidia_smi.nvmlDeviceGetFanSpeed(h)

        return info_changing
    
    def __del__(self):
        nvidia_smi = self.nvidia_smi
        nvidia_smi.nvmlShutdown()  ## This stops the ability to get device info
        super().__del__()


class CPU_Device_Checker(_Device_Checker_Base):
    """
    Class for checking CPU utilization.
    """
    def __init__(self, verbose=1):
        """
        Initialize CPU_Device_Checker class.
        """
        super().__init__(verbose=verbose)

        self.info_static = {}
        
        self.info_static['cpu_count'] = psutil.cpu_count()
        self.info_static['cpu_freq'] = psutil.cpu_freq()
        
        self.info_static['memory_total'] = psutil.virtual_memory().total
        
        self.info_static['disk_total'] = psutil.disk_usage('/').total

    def check_utilization(self):
        """
        Retrieves current utilization info from device.
        Includes: current time, memory, power, processor utilization, network utilization, and disk utilization.
        """
        info_changing = {}
        
        info_changing['time'] = datetime.datetime.now()
        
        ## log cpu utilization (per cpu), memory utilization, network utilization, disk utilization, etc
        info_changing['memory_used_percentage'] = psutil.virtual_memory().percent
        info_changing['memory_used'] = psutil.virtual_memory().used
        info_changing['memory_free'] = psutil.virtual_memory().free
        info_changing['memory_available'] = psutil.virtual_memory().available
        info_changing['memory_active'] = psutil.virtual_memory().active
        info_changing['memory_inactive'] = psutil.virtual_memory().inactive
        info_changing['memory_buffers'] = psutil.virtual_memory().buffers
        info_changing['memory_cached'] = psutil.virtual_memory().cached
        info_changing['memory_shared'] = psutil.virtual_memory().shared
        ## Get network info: current bytes sent and received
        info_changing['network_sent'] = psutil.net_io_counters().bytes_sent
        info_changing['network_received'] = psutil.net_io_counters().bytes_recv
        ## Get disk info: free space and used space and percentage
        info_changing['disk_free'] = psutil.disk_usage('/').free
        info_changing['disk_used'] = psutil.disk_usage('/').used
        info_changing['disk_used_percentage'] = psutil.disk_usage('/').percent
        ## Get disk read/write info
        info_changing['disk_read'] = psutil.disk_io_counters().read_bytes
        info_changing['disk_write'] = psutil.disk_io_counters().write_bytes
        ## Get processor info: current processor utilization (overall and per core)
        info_changing['processor_used_percentage'] = psutil.cpu_percent()
        for i_core, val in enumerate(psutil.cpu_percent(percpu=True)):
            info_changing[f'cpu_{i_core}'] = val

        return info_changing


class _RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

