## setup.py file for roicat
from pathlib import Path
import copy
import platform

from distutils.core import setup

## Get the parent directory of this file
dir_parent = Path(__file__).parent

## Get requirements from requirements.txt
def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements
deps_all = read_requirements()


## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_dict = dict(zip(deps_names, deps_all))
deps_all_latest = dict(zip(deps_names, deps_names))


# Operating system specific dependencies
# OpenCV >= 4.9 is not supported on macOS < 12
system, version_macos = platform.system(), platform.mac_ver()[0]
print(f"System: {system}")
if (system == "Darwin"):
    # Safely convert version string components to integers
    version_parts = version_macos.split('.')
    version_major_macos = int(version_parts[0])

    # Check macOS version and adjust the OpenCV version accordingly
    if (version_major_macos < 12) and ('opencv_contrib_python' in deps_all_dict):
        version_opencv_macos_sub12 = "opencv_contrib_python<=4.8.1.78"
        print(f"Detected macOS version {version_major_macos}, which is < 12. Installing an older version of OpenCV: {version_opencv_macos_sub12}")
        deps_all_dict['opencv_contrib_python'] = version_opencv_macos_sub12
        deps_all_latest['opencv_contrib_python'] = version_opencv_macos_sub12
import re
## find the numbers in the string
version_opencv = '.'.join(re.findall(r'[0-9]+', deps_all_dict['opencv_contrib_python']))
if len(version_opencv) > 0:
    version_opencv = f"<={version_opencv}"

## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = {dep: deps_all_dict[dep] for dep in [
    'numpy',
    'scipy',
    'kornia',
    'matplotlib',
    'numba',
    'scikit_learn',
    'tqdm',
    'h5py',
    'opencv_contrib_python',
    'opt_einsum',
    'optuna',
    'optuna_integration',
    'sparse',
    'natsort',
    'paramiko',
    'pandas',
    'psutil',
    'pytest',
    'hypothesis',
    'PyYAML',
    'tensorly',
    'torch',
    'torchvision',
    'torchaudio',
    'ipywidgets',
    'eva_decord',
    'wandb',
    'sqlalchemy',
    'pymysql',
    'toolz',
]}

deps_advanced = {dep: deps_all_dict[dep] for dep in [
    'tables',
    # 'rolling_quantiles',
    'pulp',
    'spconv',
    'torch_sparse',
    'av',
    'pynwb',
    'sendgrid',
    'pycuda',
    'cuml',
    'cupy',
    'cudf',
    'scanimage_tiff_reader',
    'jupyter',
    'PyWavelets',
    'mat73',
]}

deps_core_latest = {dep: deps_all_latest[dep] for dep in deps_core.keys()}


## Make versions with cv2 headless (for servers)
deps_core_dict_cv2Headless = copy.deepcopy(deps_core)
deps_core_dict_cv2Headless['opencv_contrib_python'] = 'opencv_contrib_python_headless' + version_opencv
deps_core_latest_cv2Headless = copy.deepcopy(deps_core_latest)
deps_core_latest_cv2Headless['opencv_contrib_python'] = 'opencv_contrib_python_headless'

## Print out


extras_require = {
    'all': list(deps_all_dict.values()),
    'all_latest': list(deps_all_latest.values()),
    'core': list(deps_core.values()),
    'core_latest': list(deps_core_latest.values()),
    'core_cv2Headless': list(deps_core_dict_cv2Headless.values()),
    'core_latest_cv2Headless': list(deps_core_latest_cv2Headless.values()),
}

print(extras_require)

## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get version number
with open(str(dir_parent / "bnpm" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setup(
    name='bnpm',
    version=version,
    author='Richard Hakim',
    keywords=['data analysis', 'machine learning', 'neuroscience'],
    license='LICENSE',
    description='A library of useful modules for data analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RichieHakim/basic_neural_processing_modules',

    packages=['bnpm'],

    install_requires=[],
    extras_require=extras_require,
)