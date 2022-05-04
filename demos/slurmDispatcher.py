# define job structure:
# - experiment where each iteration is the same function run on different data

# define paths:
# - load path for each iteration
# - outer save path for entire experiment
# - inner save path for each iteration

# define sbatch params 

# save files:
# - config file for experiment
# - sbatch template for experiment
# - config file for iteration
# - sbatch params for iteration
# - results files

from pathlib import Path
import json
import yaml
import os

cwd = Path(os.getcwd()).resolve()


import sys
sys.path.append('/n/data1/hms/neurobio/sabatini/rich/github_repos/')

from basic_neural_processing_modules import server


# # example test.py contents:

# print('hi')

# import sys
# args = sys.argv

# for arg in args:
#     print(arg)
    
    
# import sys
# path_script, path_params, save_dir = sys.argv

# import json
# with open(path_params, 'r') as f:
#     params = json.load(f)
    
# from pathlib import Path
# with open(Path(save_dir).resolve() / 'IT_WORKED.json', 'w') as f:
#     for param in params:
#         json.dump(param, f)



sbatch_config_default = \
"""#!/usr/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=/home/rh183/script_logs/python_01_%j.log
#SBATCH --partition=priority
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=1GB
#SBATCH --time=0-00:00:10

python "$@"
"""

script_paths = ['/n/data1/hms/neurobio/sabatini/rich/github_repos/test.py']
params_list = [[3,4], [5,6]]
sbatch_config_list = [sbatch_config_default]
max_n_jobs=2
save_dir='/n/data1/hms/neurobio/sabatini/rich/analysis/'
save_name='jobNum_'

server.batch_run(script_paths=script_paths,
                    params_list=params_list,
                    sbatch_config_list=sbatch_config_list,
                    max_n_jobs=2,
                    save_dir=save_dir,
                    save_name='jobNum_',
                    verbose=True,
                    )