from pathlib import Path
import os
import json
from . import indexing

def batch_run(script_paths, 
                params_list, 
                sbatch_config_list, 
                max_n_jobs=2,
                save_dir='/n/data1/hms/neurobio/sabatini/rich/analysis/', 
                save_name='jobNum_', 
                verbose=True,
                ):
    r"""
    Run a batch of jobs.
    Workflow 1: run a single script over a sweep of parameters
        - Make a script that takes in the set of parameters
           you wish to sweep over as variables.
        - Prepend the script to take in string arguments
           pointing to a param_config file (maybe a dict).
           See script_paths Arg below for details.
        - Save the script in .py file.
        - In a new script, call this function (batch_run)
        - A new job will be run for each item in params_list
            - Each job will make a new directory, and within
               it will save (1) a .json file containing the 
               parameters used, and (2) the .sh file that 
               was run.
        - Save output files using the 'save_dir' argument

    Alternative workflows where you have multiple different
     scripts or different config files are also possible.

    RH 2021

    Args:
        script_paths (List):
            - List of script paths to run.
            - List can contain either 1 or n_jobs items.
            - Each script must save its results it's own way
               using a relative path (see 'save_dir' below)
            - Each script should contain the following to handle
               input arguments specified by the user and this
               function, DEMO:
                ```
                import sys
                    path_script, path_params, save_dir = sys.argv
                
                import json
                with open(path_params, 'r') as f:
                    params = json.load(f)
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

                    python "$@" '
        max_n_jobs (int):
            - Maximum number of jobs that can be called
            - Used as a safety precaution
            - Be careful that params_list has the right len
        save_dir (str or Path):
            - Outer directory to save results to.
            - Will be created if it does not exist.
            - Will be populated by folders for each job
            - Will be sent to the script for each job as the
               third argument. See script_paths demo for details.
        save_name (str or List):
            - Name of each job (used as inner directory name)
            - If str, then will be used for all jobs 
            - Job iteration always appended to the end.
            - If List, then must have len(params_list) items.
        verbose (bool):
            - Whether or not to print progress
    """

    # make sure the arguments are matched in length
    n_jobs = max(len(script_paths), len(params_list), len(sbatch_config_list))
    if max_n_jobs is not None:
        if n_jobs > max_n_jobs:
            raise ValueError(f'Too many jobs requested: max_n_jobs={n_jobs} > n_jobs={max_n_jobs}')

    def rep_inputs(item, n_jobs):
        if len(item)==1 and (n_jobs>1):
            return indexing.lazy_repeat_item(item[0], pseudo_length=n_jobs)
        else:
            return item

    script_paths       = rep_inputs(script_paths,   n_jobs)
    params_list        = rep_inputs(params_list,  n_jobs)
    sbatch_config_list = rep_inputs(sbatch_config_list, n_jobs)
    save_name          = rep_inputs([save_name], n_jobs)

    # setup the save path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_dir = Path(save_dir).resolve()

    # run the jobs
    for ii in range(n_jobs):
        save_dir_job = save_dir / f'{save_name[ii]}{ii}'
        save_dir_job.mkdir(parents=True, exist_ok=True)
        # save the shell scripts
        save_path_sbatchConfig = save_dir_job / 'sbatch_config.sh'
        with open(save_path_sbatchConfig, 'w') as f:
            f.write(sbatch_config_list[ii])
        # save the parameters        
        save_path_params = save_dir_job / 'params.json'
        with open(save_path_params, 'w') as f:
            json.dump(params_list[ii], f)
    
        # run the job
        if verbose:
            print(f'Submitting job: {save_name[ii]} {ii}')
        # ! sbatch --job-name=${save_name}_${ii} --output=${save_dir_job}/log.txt --error=${save_dir_job}/err.txt --time=${sbatch_config_list[ii]["time"]} --mem=${sbatch_config_list[ii]["mem"]} --cpus-per-task=${sbatch_config_list[ii]["cpus"]} --wrap="${script_paths[ii]} ${params_list[ii]} ${sbatch_config_list[ii]} ${save_dir_job}"
        # with open()
        os.system(f'sbatch {save_path_sbatchConfig} {script_paths[ii]} {save_path_params} {save_dir_job}')
