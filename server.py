from pathlib import Path
import os
import json
import copy
import time

from . import container_helpers

def batch_run(paths_scripts, 
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

    # make sure the arguments are matched in length
    n_jobs = max(len(paths_scripts), len(params_list), len(sbatch_config_list))
    if max_n_jobs is not None:
        if n_jobs > max_n_jobs:
            raise ValueError(f'Too many jobs requested: max_n_jobs={n_jobs} > n_jobs={max_n_jobs}')

    def rep_inputs(item, n_jobs):
        if len(item)==1 and (n_jobs>1):
            return container_helpers.lazy_repeat_item(item[0], pseudo_length=n_jobs)
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
        # save the parameters        
        save_path_params = dir_save_job / 'params.json'
        with open(save_path_params, 'w') as f:
            json.dump(params_list[ii], f)
    
        # run the job
        if verbose:
            print(f'Submitting job: {name_save[ii]} {ii}')
        # ! sbatch --job-name=${name_save}_${ii} --output=${dir_save_job}/log.txt --error=${dir_save_job}/err.txt --time=${sbatch_config_list[ii]["time"]} --mem=${sbatch_config_list[ii]["mem"]} --cpus-per-task=${sbatch_config_list[ii]["cpus"]} --wrap="${paths_scripts[ii]} ${params_list[ii]} ${sbatch_config_list[ii]} ${dir_save_job}"
        # with open()
        os.system(f'sbatch {save_path_sbatchConfig} {paths_scripts[ii]} {save_path_params} {dir_save_job}')




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
        import paramiko

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
        port=22
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
        self.client.connect(hostname=hostname, username=username, password=password, port=port, look_for_keys=False, allow_agent=False)
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
    
    def receive(self, timeout=None, verbose=None):
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
        
        out = self.ssh.recv(self.nbytes).decode('utf-8')
        if verbose is None:
            verbose=self.verbose
        if verbose:
            print(out)
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
        Send a command to the remote server,
         and receive the response.
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
    
    def expect(
        self,
        str_success='(base) [rh183',
        partial_match=True,
        recv_timeout=0.3,
        total_timeout=60,
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
            
            try:
                out = self.receive(timeout=recv_timeout, verbose=verbose>0)
            except:
                if verbose==2:
                    print("expect: nothing received")
                    
            if partial_match:
                if str_success in out:
                    success = True
            else:
                if str_success == out:
                    success = True
            
            if time.time() - t_start > total_timeout:
                break
        
        if verbose==2:
            if success:
                print(f'expect succeeded')
            else:
                print(f'expect failed')
                
        return success
        
    def close(self):
        self.ssh.close()
        
    def __del__(self):
        self.ssh.close()

            
    def o2_connect(
        self,
        hostname='transfer.rc.hms.harvard.edu',
        username='rh183',
        password='',
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
        """
        self.connect(
            hostname=hostname,
            username=username,
            password=password,
            port=22
        )
        
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
        