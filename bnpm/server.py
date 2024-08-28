from pathlib import Path
import os
import json
import copy
import time
import typing
import sys
import datetime
import getpass

import natsort

from . import container_helpers

def batch_run(
    paths_scripts, 
    params_list, 
    sbatch_config_list, 
    max_n_jobs=2,
    dir_save='/n/data1/hms/neurobio/sabatini/rich/analysis/', 
    name_save='jobNum_', 
    verbose=True,
):
    r"""
    Run a batch of jobs.
    Workflow 1: run a single script over a sweep of parameters
        - Make a script that takes in the set of parameters
           you wish to sweep over as variables.
        - Prepend the script to take in string arguments
           pointing to a param_config file (maybe a dict).
           See paths_scripts Arg below for details.
        - Save the script in .py file.
        - In a new script, call this function (batch_run)
        - A new job will be run for each item in params_list
            - Each job will make a new directory, and within
               it will save (1) a .json file containing the 
               parameters used, and (2) the .sh file that 
               was run.
        - Save output files using the 'dir_save' argument

    Alternative workflows where you have multiple different
     scripts or different config files are also possible.

    RH 2021

    Args:
        paths_scripts (List):
            - List of script paths to run.
            - List can contain either 1 or n_jobs items.
            - Each script must save its results it's own way
               using a relative path (see 'dir_save' below)
            - Each script should contain the following to handle
               input arguments specified by the user and this
               function, DEMO:
                ```
                import sys
                    path_script, path_params, dir_save = sys.argv
                
                import json
                with open(path_params, 'r') as f:
                    params = json.load(f)
                ```                
            - It's also good practice to save the script .py file
               within dir_save DEMO:
                ```
                import shutil
                shutil.copy2(
                    path_script, 
                    str(Path(dir_save) / Path(path_script).name)
                    );
                ```
        params_list (List):
            - Parameters (arguments) to be used
            - List can contain either 1 or n_jobs items.
            - Each will be saved as a .json file (so nothing too big)   
            - Will be save into each inner/job directory and the path
               will be passed to the script for each job.
        sbatch_config_list (List):
            - List of string blocks containing the arguments to 
               pass for each job/script.
            - List can contain either 1 or n_jobs items.
            - Must contain: python "$@" at the bottom (to take in 
               arguments), and raw string must have '\n' to signify
               line breaks.
               Demo: '#!/usr/bin/bash
                    #SBATCH --job-name=python_01
                    #SBATCH --output=jupyter_logs/python_01_%j.log
                    #SBATCH --partition=priority
                    #SBATCH -c 1
                    #SBATCH -n 1
                    #SBATCH --mem=1GB
                    #SBATCH --time=0-00:00:10

                    unset XDG_RUNTIME_DIR

                    cd /n/data1/hms/neurobio/sabatini/rich/

                    date

                    echo "loading modules"
                    module load gcc/9.2.0 cuda/11.2

                    echo "activating environment"
                    source activate ROI_env

                    echo "starting job"
                    python "$@" '
        max_n_jobs (int):
            - Maximum number of jobs that can be called
            - Used as a safety precaution
            - Be careful that params_list has the right len
        dir_save (str or Path):
            - Outer directory to save results to.
            - Will be created if it does not exist.
            - Will be populated by folders for each job
            - Will be sent to the script for each job as the
               third argument. See paths_scripts demo for details.
        name_save (str or List):
            - Name of each job (used as inner directory name)
            - If str, then will be used for all jobs 
            - Job iteration always appended to the end.
            - If List, then must have len(params_list) items.
        verbose (bool):
            - Whether or not to print progress
    """
    import json
    import os
    import shutil

    ## Assert that the inputs are valid
    for val in ['paths_scripts', 'params_list', 'sbatch_config_list']:
        assert isinstance(locals()[val], list), f'{val} must be a list'

    # make sure the arguments are matched in length
    n_jobs = max(len(paths_scripts), len(params_list), len(sbatch_config_list))
    if max_n_jobs is not None:
        if n_jobs > max_n_jobs:
            raise ValueError(f'Too many jobs requested: max_n_jobs={max_n_jobs} > n_jobs={n_jobs}')

    def rep_inputs(item, n_jobs):
        if len(item)==1 and (n_jobs>1):
            return container_helpers.Lazy_repeat_obj(item[0], pseudo_length=n_jobs)
        else:
            return item

    paths_scripts      = rep_inputs(paths_scripts,   n_jobs)
    params_list        = rep_inputs(params_list,  n_jobs)
    sbatch_config_list = rep_inputs(sbatch_config_list, n_jobs)
    name_save          = rep_inputs([name_save], n_jobs)

    # setup the save path
    Path(dir_save).mkdir(parents=True, exist_ok=True)
    dir_save = Path(dir_save).resolve()

    # run the jobs
    for ii in range(n_jobs):
        dir_save_job = dir_save / f'{name_save[ii]}{ii}'
        dir_save_job.mkdir(parents=True, exist_ok=True)
        # save the shell scripts
        save_path_sbatchConfig = dir_save_job / 'sbatch_config.sh'
        with open(save_path_sbatchConfig, 'w') as f:
            f.write(sbatch_config_list[ii])
        # save the script
        path_script_job = dir_save_job / Path(paths_scripts[ii]).name
        shutil.copyfile(paths_scripts[ii], path_script_job);
        # save the parameters        
        path_params_job = dir_save_job / 'params.json'
        with open(path_params_job, 'w') as f:
            json.dump(params_list[ii], f)
    
        # run the job
        if verbose:
            print(f'Submitting job: {name_save[ii]} {ii}')
        # ! sbatch --job-name=${name_save}_${ii} --output=${dir_save_job}/log.txt --error=${dir_save_job}/err.txt --time=${sbatch_config_list[ii]["time"]} --mem=${sbatch_config_list[ii]["mem"]} --cpus-per-task=${sbatch_config_list[ii]["cpus"]} --wrap="${paths_scripts[ii]} ${params_list[ii]} ${sbatch_config_list[ii]} ${dir_save_job}"
        os.system(f'sbatch {save_path_sbatchConfig} {paths_scripts[ii]} {path_params_job} {dir_save_job}')


###############################
### MASTER CONTROLLER STUFF ###
###############################

import paramiko
import time
from pathlib import Path
import os
import stat
import re

class ssh_interface():
    """
    Interface to ssh to a remote server.
    Mostly a wrapper for paramiko.SSHClient.
    Tested on O2 cluster at Harvard.
    RH 2022
    """
    def __init__(
        self,
        nbytes_toReceive=4096,
        recv_timeout=1,
        verbose=True,
    ):
        """
        Args:
            nbytes_toReceive (int):
                Number of bytes to receive at a time.
                Caps the maximum message it can receive.
            recv_timeout (int):
                Timeout for receiving data.
            verbose (bool):
                Whether or not to print progress
        """
        
        self.nbytes = nbytes_toReceive
        self.recv_timeout = recv_timeout
        self.verbose=verbose
        
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    def connect(
        self,
        hostname='transfer.rc.hms.harvard.edu',
        username='rh183',
        password='',
        port=22,
        key_filename=None,
        look_for_keys=True,
    ):
        """
        Connect to the remote server.
        Args:
            hostname (str):
                Hostname of the remote server.
            username (str):
                Username to log in with.
            password (str):
                Password to log in with.
                Is not stored.
            port (int):
                Port to connect to.
                sftp is always port 22.
        """
        self.client.connect(
            hostname=hostname,
            username=username,
            password=password,
            port=port, 
            key_filename=key_filename,
            look_for_keys=look_for_keys, 
            allow_agent=look_for_keys,
        )
        self.ssh = self.client.invoke_shell()

    def send(self, cmd='ls', append_enter=True):
        """
        Send a command to the remote server.
        Args:
            cmd (str):
                Command to send.
            append_enter (bool):
                Whether or not to append an enter
                 (backslash n) to the command.
                This is usually necessary to send.
        """
        if append_enter:
            cmd += '\n'
        self.ssh.send(cmd)
    
    def receive(self, timeout=None, verbose=True, throw_error=False):
        """
        Receive data from the remote server.
        Args:
            timeout (int):
                Timeout for receiving data.
                If None, will use self.recv_timeout.
            verbose (bool):
                Whether or not to print progress.
        """
        if timeout is None:
            timeout = self.recv_timeout
        self.ssh.settimeout(timeout)
        
        try:
            out = self.ssh.recv(self.nbytes, ).decode('utf-8');
            if verbose:
                print(out)
        except:
            if throw_error:
                raise
            else:
                if verbose:
                    print('Timeout')
                out = None
        return out
    
    def send_receive(
        self, 
        cmd='ls', 
        append_enter=True,
        post_send_wait_t=0.1, 
        timeout=None, 
        verbose=None,
    ):
        """
        Send a command to the remote server, wait for 
         some time, and then receive the output.
        Args:
            cmd (str):
                Command to send.
            append_enter (bool):
                Whether or not to append an enter
                 (backslash n) to the command.
                This is usually necessary to send.
            post_send_wait_t (float):
                Time to wait after sending the command.
            timeout (int):
                Timeout for receiving data.
                If None, will use self.recv_timeout.
            verbose (bool):
                Whether or not to print progress.
        """
        self.send(cmd=cmd, append_enter=append_enter)
        time.sleep(post_send_wait_t)
        out = self.receive(timeout=timeout, verbose=verbose)   
        return out 
    
    def expect(
        self,
        str_success='(base) [rh183',
        partial_match=True,
        recv_timeout=0.3,
        total_timeout=60,
        sleep_time=0.1,
        error_on_timeout=False,
        verbose=None,
    ):
        """
        Wait for a string to appear in the output.
        Args:
            str_success (str):
                String to wait for.
            partial_match (bool):
                Whether or not to allow a partial match.
            recv_timeout (float):
                Timeout for receiving data per 
                 check iteration.
            total_timeout (float):
                Total time to wait for the string.
            sleep_time (float):
                Time to sleep between checks.
                Allows for keyboard interrupts.
            error_on_timeout (bool):
                Whether or not to raise an error if the
                total_timeout is reached.
            verbose (bool):
                Whether or not to print progress.
                0/False: no printing
                1/True: will print recv outputs.
                2: will print expect progress.
                None: will default to self.verbose (1 or 2).
        """
        t_start = time.time()
        
        if recv_timeout is None:
            recv_timeout = self.recv_timeout
        if verbose is None:
            verbose = self.verbose
            
        success = False
        out=''
        while success is False:
            if verbose==2:
                print(f'=== expecting, t={time.time() - t_start} ===')
            
            out = self.receive(timeout=recv_timeout, verbose=False)
            if out is None:
                if verbose==2:
                    print("expect: nothing received")
            elif verbose > 0:
                print(out)
                    
            if partial_match and (out is not None):
                if str_success in out:
                    success = True
            else:
                if str_success == out:
                    success = True
            
            time.sleep(sleep_time)

            if time.time() - t_start > total_timeout:
                if error_on_timeout:
                    raise TimeoutError(f'Expect timeout: {str_success} not found in {total_timeout} seconds')
                else:
                    break
        
        if verbose==2:
            if success:
                print(f'expect succeeded')
            else:
                print(f'expect failed')
                
        return out, success

    def send_expect(
        self,
        cmd='ls',
        str_success='(base) [rh183',
        partial_match=True,
        recv_timeout=0.3,
        total_timeout=60,
        sleep_time=0.1,
        error_on_timeout=False,
        verbose=None,
    ):
        """
        Send a command to the remote server, wait for 
         a string to appear in the output, and then 
         receive the output.
        Args:
            cmd (str):
                Command to send.
            str_success (str):
                String to wait for.
            partial_match (bool):
                Whether or not to allow a partial match.
            recv_timeout (float):
                Timeout for receiving data per 
                 check iteration.
            total_timeout (float):
                Total time to wait for the string.
            sleep_time (float):
                Time to sleep between checks.
                Allows for keyboard interrupts.
            error_on_timeout (bool):
                Whether or not to raise an error if the
                total_timeout is reached.
            verbose (bool):
                Whether or not to print progress.
                0/False: no printing
                1/True: will print recv outputs.
                2: will print expect progress.
                None: will default to self.verbose (1 or 2).
        """
        self.send(cmd=cmd)
        out, success = self.expect(
            str_success=str_success,
            partial_match=partial_match,
            recv_timeout=recv_timeout,
            total_timeout=total_timeout,
            sleep_time = sleep_time,
            error_on_timeout=error_on_timeout,
            verbose=verbose,
        )
        return out, success

    def initialize_sftp(self):
        """
        Initialize the SFTP client.
        """
        self.sftp = self.client.open_sftp()
        return self.sftp
        
    def close(self):
        self.ssh.close()
        
    def __del__(self):
        if self.ssh is not None:
            self.ssh.close()

            
    def o2_connect(
        self,
        hostname='transfer.rc.hms.harvard.edu',
        username='rh183',
        password='',
        skip_passcode=False,
        key_filename=None,
        look_for_keys=False,
        passcode_method=1,
        verbose=1,
    ):
        """
        Connect to the O2 cluster.
        Helper function with some hard-coded expectations.
        Args:
            hostname (str):
                Hostname of the remote server.
            username (str):
                Username to log in with.
            password (str):
                Password to log in with.
                Is not stored.
            passcode_method (int):
                Method to use for O2 passcode.
                1. Duo Push
                2. Phone call
                3. SMS passcodes
            verbose (int):
                0/False: no printing
                1/True: will print recv outputs.
                2: will print expect progress.
                None: will default to self.verbose (1 or 2).
            skip_passcode (bool):
                Whether or not to skip the passcode step.
        """
        self.connect(
            hostname=hostname,
            username=username,
            password=password,
            port=22,
            key_filename=key_filename,
            look_for_keys=look_for_keys,
        )
        
        if skip_passcode==False:
            self.expect(
                str_success=f'Passcode or option (1-3)',
                partial_match=True,
                recv_timeout=0.3,
                total_timeout=60,
                verbose=verbose,
            )
            
            self.send(cmd=str(passcode_method))
        
        self.expect(
            str_success=f'[{username}@',
            partial_match=True,
            recv_timeout=0.3,
            total_timeout=60,
            verbose=verbose,
        )

    def fasrc_connect(
        self,
        hostname='login.rc.fas.harvard.edu',
        username='rhakim',
        password=None,
        verbose=1,
    ):
        """
        Connect to the FASRC cluster using interactive authentication. \n
        Helper function with some hard-coded expectations.
        Args:
            hostname (str):
                Hostname of the remote server.
            username (str):
                Username to log in with.
            verbose (int):
                0/False: no printing
                1/True: will print recv outputs.
                2: will print expect progress. (Sets logging level to DEBUG)
                None: will default to self.verbose (1 or 2).
        """
        def interactive_handler(title, instructions, prompt_list):
            responses = []
            for prompt, echo in prompt_list:
                if "password" in prompt.lower():
                    if password is not None:
                        responses.append(password)
                    else:
                        ## Prompt user for password
                        responses.append(getpass.getpass(prompt))
                elif "verification" in prompt.lower():
                    responses.append(getpass.getpass(prompt))
                else:
                    print(f"Prompt not recognized: {prompt}")
            return responses
        
        if verbose > 1:
            import logging
            logging.basicConfig(level=logging.DEBUG)

        ## Below try statement is needed in order to set up the client attributes. 
        ## The easiest way is to just try to connect (and set attrs), fail and move on.
        try:
            self.client.connect(hostname=hostname, allow_agent=False, look_for_keys=False)
        except:
            pass
        
        transport = self.client.get_transport()
        transport.auth_interactive(username=username, handler=interactive_handler)
        self.ssh = self.client.invoke_shell()
        self.expect(
            str_success=f'{username}@',
            partial_match=True,
            recv_timeout=0.3,
            total_timeout=60,
            verbose=verbose,
        )

class sftp_interface():
    """
    Interface to sftp with a remote server.
    Mostly a wrapper for paramiko.SFTPClient.
    Tested on O2 cluster at Harvard.
    RH 2022
    """
    def __init__(
        self,
        ssh_client=None,
        hostname="transfer.rc.hms.harvard.edu",
        port=22,
    ):
        """
        Args:
            ssh_obj (paramiko.SSHClient):
                SSHClient object to use.
                Can be taken from ssh_interface with
                 ssh_interface.client
            hostname (str):
                Hostname of the remote server.
            port (int):
                Port of the remote server.
        """
        if ssh_client is None:
            self.transport = paramiko.Transport((hostname, port))  ## open a transport object
        else:
            if isinstance(ssh_client, ssh_interface):
                self.client = ssh_client
                self.sftp = self.client.open_sftp()
            elif isinstance(ssh_client, paramiko.SSHClient):
                self.client = ssh_client
                self.sftp = self.client.open_sftp()
            elif isinstance(ssh_client, paramiko.Channel):
                self.transport = ssh_client.get_transport()
                self.sftp = paramiko.SFTPClient.from_transport(self.transport)
                self.client = None
        
    def connect(
        self,
        username='rh183',
        password=''
    ):
        """
        Connect to the remote server.
        Args:
            username (str):
                Username to log in with.
            password (str):
                Password to log in with.
                Is not stored.
        """
        if hasattr(self, 'transport'):
            self.transport.connect(None, username, password)  ## authorization
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)  ## open sftp
        elif hasattr(self, 'client'):
            self.sftp = self.client.open_sftp()
        else:
            raise ValueError('No valid connection method found')
        
        self.ssh = ssh_interface(ssh_client=self.client)
    
    def put_dir(self, source, target, mkdir=True, verbose=True):
        '''
        Uploads the contents of the source directory to the target path.
        All subdirectories in source are created under target recusively.
        Args:
            source (str):
                Path to the source directory (local).
            target (str):
                Path to the target directory (remote).
            mkdir (bool):
                Make the target directory if it does not exist.
        '''
        source = Path(source).resolve()
        target = Path(target).resolve()
        
        if mkdir:
            self.mkdir_safe(str(target), ignore_existing=True)

        for item in os.listdir(source):
            if os.path.isfile(source / item):
                if verbose:
                    print(f'uploading {source / item}   to   {target / item}')
                self.sftp.put(str(source / item) , str(target / item))
            else:
                self.mkdir_safe(str(target / item) , ignore_existing=True)
                self.put_dir(source / item , target / item)

    def get_dir(self, source, target, mkdir=True, verbose=True, prog_bars=False):
        '''
        Downloads the contents of the source directory to the target path. All
        subdirectories in source are created under target recusively.
        
        Args:
            source (str):
                Path to the source directory (remote).
            target (str):
                Path to the target directory (local).
            mkdir (bool):
                Whether or not to create the target directory.
            verbose (bool):
                Whether or not to print progress of files being downloaded.
        '''
        if mkdir:
            Path(target).mkdir(parents=True, exist_ok=True)
        source = Path(source).resolve()
        target = Path(target).resolve()
        
        for item in self.sftp.listdir(str(source)):
            if self.isdir_remote(str(source / item)):
                (target / item).mkdir(parents=True, exist_ok=True)  ## probably not necessary anymore since done above
                self.get_dir(source / item , target / item)
            else:
                if verbose:
                    print(f'downloading {source / item}   to   {target / item}')
                self.get(
                    remotepath=str(source / item), 
                    localpath=str(target / item),
                    mkdirs=mkdir,
                    prefetch=False,
                    prog_bar=prog_bars,
                )
        
    def mkdir_safe(self, path_remote, mode=511, ignore_existing=False):
        '''
        Augments mkdir by adding an option to not fail if the folder exists.
        Will fail if the outer directory does not exist.
        Args:
            path_remote (str):
                Path to the remote directory.
            mode (int):
                Mode to set the directory to.   
            ignore_existing (bool):
                Whether or not to ignore existing folders.
                If True, will not fail if the folder already exists.
                If False, will fail if the folder already exists.
        '''
        try:
            self.sftp.mkdir(path_remote, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
    
    def mkdir_p(self, dir_remote):
        """
        Change to this directory, recursively making new folders if needed.
        Returns True if any folders were created.
        Args:
            dir_remote (str):
                Path to the remote directory.
        """
        if dir_remote == '/':
            # absolute path so change directory to root
            self.sftp.chdir('/')
            return
        if dir_remote == '':
            # top-level relative directory must exist
            return
        try:
            self.sftp.chdir(dir_remote) # sub-directory exists
        except IOError:
            dirname, basename = os.path.split(dir_remote.rstrip('/'))
            self.mkdir_p(dirname) # make parent directories
            self.sftp.mkdir(basename) # sub-directory missing, so created it
            self.sftp.chdir(basename)
            return True
    
    def isdir_remote(self, path):
        """
        Checks if a remote path is a directory.
        Args:
            path (str):
                Path to the remote directory.
        """
        try:
            return stat.S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            #Path does not exist, so by definition not a directory
            return False

    def search_recursive(
        self, 
        dir_outer='.', 
        reMatch='', 
        reMatch_in_path='',
        depth=6,
        find_files=True,
        find_folders=True,
        natsorted=True,
        verbose=True
    ):
        """
        Searches a remote directory recursively for files or directories
         matching a pattern.
        Args:
            sftp (paramiko.SFTPClient):
                SFTPClient object.
            dir_outer (str):
                Current working directory.
            reMatch (str):
                Regular expression to search for.
            reMatch_in_path (str):
                Regular expression to search for in the path.
            depth (int):
                Maximum depth (number of hierarchical subdirectories) to search.
            find_files (bool):
                Whether or not to search for files.
            find_folders (bool):
                Whether or not to search for folders.
            natsorted (bool):
                Whether or not to return the results natsorted.
            verbose (bool):
                Whether or not to print the paths of the files found.
                If False, no output is printed.
                If True or 1, prints the paths of the files found.
                If >1, prints the current search directory.

        Returns:
            list:cod
                List of paths to the files found.
        """
        search_results = []

        def _recursive_search(search_results, sftp, cwd='.', depth_current=0, verbose=True):
            if depth_current > depth:
                return search_results
            contents = {name: stat.S_ISDIR(attr.st_mode)  for name, attr in zip(sftp.listdir(cwd), sftp.listdir_attr(cwd))}
            for name, isdir  in contents.items():
                if (isdir and find_folders) or (not isdir and find_files):
                    if re.search(reMatch, name) and re.search(reMatch_in_path, str(Path(cwd) / name)):
                        path_found = str(Path(cwd) / name)
                        search_results.append(path_found)
                        print(path_found) if verbose else None

                if isdir:
                    search_results = _recursive_search(
                        search_results=search_results,
                        sftp=sftp, 
                        cwd=str(Path(cwd) / name), 
                        depth_current=depth_current+1,
                        verbose=verbose
                    )

                print(f'cwd: {cwd}, name: {name}, isdir: {isdir}, depth: {depth_current}') if verbose > 1 else None
            return search_results
        
        fn_sort = natsort.natsorted if natsorted else lambda x: x
        
        return fn_sort(_recursive_search(
            search_results, 
            self.sftp, 
            cwd=dir_outer, 
            depth_current=0, 
            verbose=verbose,
        ))
        
    def list_fileSizes_recursive(self, directory='.'):
        """
        Lists the sizes of all files in a remote directory recursively.
        Args:
            sftp (paramiko.SFTPClient):
                SFTPClient object.
            directory (str):
                directory to search within.

        Returns:
            dict_of_sizes:
                Dictionary of paths and sizes of files
                 {'path_of_file', size) of all files found.
        """
        sizes = []

        def _recursive_list_sizes(sftp, cwd='.', sizes=[]):
            contents = {name: stat.S_ISDIR(attr.st_mode)  for name, attr in zip(sftp.listdir(cwd), sftp.listdir_attr(cwd))}
            for name, isdir  in contents.items():
                if isdir:
                    sizes = _recursive_list_sizes(sftp, str(Path(cwd) / name), sizes=sizes)
                else:
                    size = sftp.stat(str(Path(cwd) / name)).st_size
                    sizes.append((str(Path(cwd) / name), size))
            return sizes

        return dict(_recursive_list_sizes(self.sftp, cwd=directory, sizes=sizes))

    
    def get_fileProperties(self, paths, error_on_missing=False):
        """
        Return a dictionary of properties for one or more files.

        Args:
            paths (str or list of str):
                Path or list of paths to the files.
            error_on_missing (bool):
                Whether or not to raise an error if a file is missing.
        Returns:
            dict:
                Dictionary of properties for each file.
        """
        if isinstance(paths, str):
            paths = [paths]
        
        props_raw = {}
        for path in paths:
            try:
                props_raw[path] = self.sftp.stat(path)
            except FileNotFoundError:
                if error_on_missing:
                    raise FileNotFoundError(f'File not found: {path}')
                else:
                    props_raw[path] = None

        props = {}
        for path, prop in props_raw.items():
            if prop is not None:
                props[path] = {
                    'size': prop.st_size,
                    'last_modified': datetime.datetime.fromtimestamp(prop.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'last_accessed': datetime.datetime.fromtimestamp(prop.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
                    'permissions': prop.st_mode,
                }
            else:
                props[path] = None
        
        if len(props) == 1:
            props = props[paths[0]]

        return props


    def hierarchical_folderSizes(self, directory='.'):
        """
        Makes a hierarchical dict of dicts containing the 
         directory tree structure and sizes of each folder.
        Args:
            sftp (paramiko.SFTPClient):
                SFTPClient object.
            directory (str):
                Current working directory.

        Returns:
            h_dict:
                Hierarchical dict of dicts with fields
                 {name: children, '_size': size} where children contains
                 either dictionaries for children.
        """

        def _recursive_sum_sizes(sftp, cwd='.', size=0):
            contents = {name: {} for name in sftp.listdir(cwd)}
            isdir = [stat.S_ISDIR(attr.st_mode) for attr in sftp.listdir_attr(cwd)]
            for (name,v), isd in zip(list(contents.items()), isdir):
                if isd==False:  ## file
                    contents[name]['_size'] = sftp.stat(str(Path(cwd) / name)).st_size
                else:                     ## directory
                    contents[name], contents[name]['_size'] = _recursive_sum_sizes(sftp, cwd=str(Path(cwd) / name), size=0)
                size += contents[name]['_size']
            return contents, size

        out = _recursive_sum_sizes(self.sftp, cwd=directory, size=0)
        ret = {directory: out[0]}
        ret[directory]['_size'] = out[1]
        return ret

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def get(
        self, 
        remotepath, 
        localpath, 
        mkdirs=True,
        prefetch=False, 
        callback=None, 
        prog_bar=True,
    ):
        """
        Get a file from the remote host.
        Uses a fancy class to do the downloading. It doesn't slow
         down due to the size of the file.
        Some extra stuff in here like a progress bar.
        RH 2022

        Args:
            remote_path (str):
                Path to the file on the remote host.
            local_path (str):
                Path to the file on the local host.
            prefetch (bool):
                Whether or not to prefetch the file.
            callback (function):
                Callback function to call within self.sftp.get
            prog_bar (bool):
                Whether or not to show a progress bar.
        
        Returns:
            None
        """
        Path(localpath).parent.mkdir(parents=True, exist_ok=True) if mkdirs else None

        with _TqdmWrap(ascii=False, unit='b', unit_scale=True) as pbar:
            def conj_func(a,b):
                pbar.viewBar(a, b)
                if callback is not None:
                    callback(a, b)
                    
            with self.sftp.open(remotepath, 'rb') as f_in, open(localpath, 'wb') as f_out:
                _SFTPFileDownloader(
                    f_in=f_in,
                    f_out=f_out,
                    prefetch=prefetch,
                    callback=conj_func if prog_bar else callback,
                ).download()

    def compare_directories(
        self,
        dir_remote,
        dir_local,
        depth=None,
        verbose=True,
    ):
        """
        Compares a local directory with a remote directory using file metadata.

        Args:
            dir_remote (str):
                Path to the remote directory.
            dir_local (str):
                Path to the local directory.
            depth (int):
                Maximum depth to search. If None, will search recursively
                through all subdirectories and files. If 0, will only search the
                current directory.
            verbose (bool):
                Whether or not to print progress: \n
                    * 0/False: no printing \n
                    * 1/True: will print found files \n
                    * 2: will print the current directory being searched.
        """
        print(f"Walking remote directory: {dir_remote}") if verbose else None
        paths_rel_remote = [str(Path(p).relative_to(dir_remote)) for p in self.walk_remote(dir_remote, depth=depth, verbose=verbose > 1)]
        print(f"Walking local directory: {dir_local}") if verbose else None
        paths_rel_local = [str(Path(p).relative_to(dir_local)) for p in self.walk_local(dir_local, depth=depth, verbose=verbose > 1)]

        paths_common = set(paths_rel_remote) & set(paths_rel_local)
        paths_unique_remote = set(paths_rel_remote) - set(paths_rel_local)
        paths_unique_local = set(paths_rel_local) - set(paths_rel_remote)

        def _metadata_remote(path):
            stat = self.sftp.stat(str(Path(dir_remote) / path))
            return {
                'size': stat.st_size,
                'last_modified': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'permissions': stat.st_mode,
            }
        def _metadata_local(path):
            stat = (Path(dir_local) / path).stat()
            return {
                'size': stat.st_size,
                'last_modified': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'permissions': stat.st_mode,
            }

        metadata_remote = {path: _metadata_remote(path) for path in paths_common | paths_unique_remote}
        metadata_local = {path: _metadata_local(path) for path in paths_common | paths_unique_local}

        
        metadata_matched = {path: {meta: metadata_remote[path][meta] == metadata_local[path][meta] for meta in metadata_remote[path]} for path in paths_common}

        return {
            'paths_common': paths_common,
            'paths_unique_remote': paths_unique_remote,
            'paths_unique_local': paths_unique_local,
            'metadata_remote': metadata_remote,
            'metadata_local': metadata_local,
            'metadata_matched': metadata_matched,
        }

    def walk_remote(
        self,
        dir_remote,
        depth=None,
        verbose=True,
    ):
        """
        Walks through a remote directory. Yielding the path to each file. Does
        not return empty directories.
        Args:
            dir_remote (str):
                Path to the remote directory.
            depth (int):
                Maximum depth to search. If None, will search recursively
                through all subdirectories and files. If 0, will only search the
                current directory.
            verbose (bool):
                Whether or not to print progress: \n
                    * 0/False: no printing \n
                    * 1/True: will print found files \n
                    * 2: will print the current directory being searched.
        """
        if depth is None:
            depth = float('inf')
        
        def _recursive_walk(sftp, cwd='.', depth_current=0, verbose=True):
            if depth_current > depth:
                return
            contents = {name: stat.S_ISDIR(attr.st_mode)  for name, attr in zip(sftp.listdir(cwd), sftp.listdir_attr(cwd))}
            for name, isdir in contents.items():
                if isdir:
                    print(f'cwd: {cwd}, name: {name}, isdir: {isdir}, depth: {depth_current}') if verbose > 1 else None
                    yield from _recursive_walk(sftp, cwd=str(Path(cwd) / name), depth_current=depth_current+1, verbose=verbose)
                else:
                    print(f"found: {str(Path(cwd) / name)}") if verbose else None
                    yield str(Path(cwd) / name)
                    
        return _recursive_walk(self.sftp, cwd=dir_remote, depth_current=0, verbose=verbose)
    
    def walk_local(
        self,
        dir_local,
        depth=None,
        verbose=True,
    ):
        """
        Walks through a local directory. Yielding the path to each file. Does
        not return empty directories.
        Args:
            dir_local (str):
                Path to the local directory.
            depth (int):
                Maximum depth to search. If None, will search recursively
                through all subdirectories and files. If 0, will only search the
                current directory.
            verbose (bool):
                Whether or not to print progress: \n
                    * 0/False: no printing \n
                    * 1/True: will print found files \n
                    * 2: will print the current directory being searched.
        """
        if depth is None:
            depth = float('inf')
        
        def _recursive_walk(cwd, depth_current=0, verbose=True):
            if depth_current > depth:
                return
            contents = {name: Path(cwd / name).is_dir() for name in os.listdir(cwd)}
            for name, isdir in contents.items():
                if isdir:
                    print(f'cwd: {cwd}, name: {name}, isdir: {isdir}, depth: {depth_current}') if verbose > 1 else None
                    yield from _recursive_walk(cwd / name, depth_current=depth_current+1, verbose=verbose)
                else:
                    print(f"found: {cwd / name}") if verbose else None
                    yield str(cwd / name)
                    
        return _recursive_walk(Path(dir_local), depth_current=0, verbose=verbose)

    def close(self):
        if hasattr(self, 'transport'):
            self.transport.close()
        elif hasattr(self, 'client'):
            self.client.close()
        elif hasattr(self, 'sftp'):
            self.sftp.close()

    def refresh_connection(self):
        self.connect()

def make_rsync_command(
    source, 
    destination, 
    recursive=True, 
    preserve_attributes=True, 
    verbose=True, 
    dry_run=False,
    delete=False, 
    compress=False,
    archive=False,
    exclude=None, 
    include=None, 
    extra_options=None,
):
    """
    Constructs an rsync command based on the provided options.
    RH 2023

    Args:
        source (str): 
            The source path for the rsync command.
        destination (str): 
            The destination path for the rsync command.
        recursive (bool, optional): 
            If True, transfer directories recursively. This means that
            directories will be copied along with their contents.
        preserve_attributes (bool, optional): 
            If True, preserve file attributes. Like: permissions, symbolic
            links, etc.
        verbose (bool, optional): 
            If True, enable verbose output. 
        dry_run (bool, optional): 
            If True, perform a trial run without changes. 
        delete (bool, optional): 
            If True, delete files in the destination that are not in source.
            Ensures that the destination is an exact copy of the source.
        compress (bool, optional): 
            If True, compress file data during transfer. 
        archive (bool, optional): 
            If True, use archive mode (preserves permissions, symbolic links,
            etc.). 
        exclude (list[str], optional): 
            List of patterns to exclude. 
        include (list[str], optional): 
            List of patterns to include. .
        extra_options (list[str], optional): 
            Additional rsync options to include. 

    Returns:
        str: 
            The constructed rsync command.

    Example:
        ``` make_rsync_command("/path/to/source", "/path/to/dest", delete=True)
        "rsync -r -p -v --delete /path/to/source /path/to/dest" ```
    """

    # Initialize exclude and include if they're None
    if exclude is None:
        exclude = []
    if include is None:
        include = []

    # Start with the rsync command
    cmd = ["rsync"]
    
    # Include options based on function arguments
    if archive:
        cmd.append("-a")
    else:
        if recursive:
            cmd.append("-r")
        if preserve_attributes:
            cmd.append("-p")
    if verbose:
        cmd.append("-v")
    if dry_run:
        cmd.append("--dry-run")
    if delete:
        cmd.append("--delete")
    if compress:
        cmd.append("-z")
    for pattern in exclude:
        cmd.extend(["--exclude", pattern])
    for pattern in include:
        cmd.extend(["--include", pattern])
    if extra_options:
        cmd.extend(extra_options)
    
    # Add source and destination
    cmd.extend([source, destination])

    return ' '.join(cmd)


"""
Implemented by casperdcl here: https://github.com/tqdm/tqdm/issues/311#issuecomment-387066847
"""
try:
    from tqdm.auto import tqdm
except ImportError:
    class _TqdmWrap(object):
        # tqdm not installed - construct and return dummy/basic versions
        def __init__(self, *a, **k):
            pass

        def viewBar(self, a, b):
            # original version
            res = a / int(b) * 100
            sys.stdout.write('\rComplete precent: %.2f %%' % (res))
            sys.stdout.flush()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
else:
    class _TqdmWrap(tqdm):
        def viewBar(self, a, b):
            self.total = int(b)
            self.update(int(a - self.n))  # update pbar with increment


class _SFTPFileDownloader:
    """
    Helper class to download large file with paramiko sftp client with limited number of concurrent requests.
    Implemented by vznncv here: https://gist.github.com/vznncv/cb454c21d901438cc228916fbe6f070f
    """

    _DOWNLOAD_MAX_REQUESTS = 48
    _DOWNLOAD_MAX_CHUNK_SIZE = 0x8000

    def __init__(self, f_in: paramiko.SFTPFile, f_out: typing.BinaryIO, callback=None, prefetch=False):
        self.f_in = f_in
        self.f_out = f_out
        self.callback = callback
        self.prefetch = prefetch

        self.requested_chunks = {}
        self.received_chunks = {}
        self.saved_exception = None

    def download(self):
        file_size = self.f_in.stat().st_size
        requested_size = 0
        received_size = 0

        while True:
            # send read requests
            while len(self.requested_chunks) + len(self.received_chunks) < self._DOWNLOAD_MAX_REQUESTS and \
                    requested_size < file_size:
                chunk_size = min(self._DOWNLOAD_MAX_CHUNK_SIZE, file_size - requested_size)
                request_id = self._sftp_async_read_request(
                    fileobj=self,
                    file_handle=self.f_in.handle,
                    offset=requested_size,
                    size=chunk_size
                )
                self.requested_chunks[request_id] = (requested_size, chunk_size)
                requested_size += chunk_size

            # receive blocks if they are available
            # note: the _async_response is invoked
            self.f_in.sftp._read_response()
            self._check_exception()

            # write received data to output stream
            while True:
                chunk = self.received_chunks.pop(received_size, None)
                if chunk is None:
                    break
                _, chunk_size, chunk_data = chunk
                self.f_out.write(chunk_data)
                if self.callback is not None:
                    self.callback(received_size, file_size)

                received_size += chunk_size

            # check transfer status
            if received_size >= file_size:
                break

            # check chunks queues
            if not self.requested_chunks and len(self.received_chunks) >= self._DOWNLOAD_MAX_REQUESTS:
                raise ValueError("SFTP communication error. The queue with requested file chunks is empty and"
                                 "the received chunks queue is full and cannot be consumed.")

        return received_size

    def _sftp_async_read_request(self, fileobj, file_handle, offset, size):
        sftp_client = self.f_in.sftp

        with sftp_client._lock:
            num = sftp_client.request_number

            msg = paramiko.Message()
            msg.add_int(num)
            msg.add_string(file_handle)
            msg.add_int64(offset)
            msg.add_int(size)

            sftp_client._expecting[num] = fileobj
            sftp_client.request_number += 1

        sftp_client._send_packet(paramiko.sftp.CMD_READ, msg)
        return num

    def _async_response(self, t, msg, num):
        if t == paramiko.sftp.CMD_STATUS:
            # save exception and re-raise it on next file operation
            try:
                self.f_in.sftp._convert_status(msg)
            except Exception as e:
                self.saved_exception = e
            return
        if t != paramiko.sftp.CMD_DATA:
            raise paramiko.SFTPError("Expected data")
        data = msg.get_string()

        chunk_data = self.requested_chunks.pop(num, None)
        if chunk_data is None:
            return

        # save chunk
        offset, size = chunk_data

        if size != len(data):
            raise paramiko.SFTPError(f"Invalid data block size. Expected {size} bytes, but it has {len(data)} size")
        self.received_chunks[offset] = (offset, size, data)

    def _check_exception(self):
        """if there's a saved exception, raise & clear it"""
        if self.saved_exception is not None:
            x = self.saved_exception
            self.saved_exception = None
            raise x


def pw_encode(pw):
    import base64
    if pw is not None:
        return base64.b64encode(pw.encode("utf-8"))
    else:
        return None
def pw_decode(pw):
    import base64
    if pw is not None:
        return base64.b64decode(pw).decode("utf-8")
    else:
        return None
        