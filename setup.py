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


## Operating system specific dependencies
### OpenCV >= 4.9 is not supported on macOS < 12
system, version = platform.system(), platform.mac_ver()[0]
if system == "Darwin" and version and ('opencv_contrib_python' in deps_all_dict):
    if tuple(map(int, version.split('.'))) < (12, 0, 0):
        version_opencv_macosSub12 = "opencv_contrib_python<=4.8.1.78"
        deps_all_dict['opencv_contrib_python'], deps_all_latest['opencv_contrib_python'] = version_opencv_macosSub12, version_opencv_macosSub12


## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = {dep: deps_all_dict[dep] for dep in [
    'numpy',
    'scipy',
    'kornia',
    'matplotlib',
    'numba',
    'scikit-learn',
    'tqdm',
    'h5py',
    'opencv-contrib-python',
    'optuna',
    'sparse',
    'natsort',
    'paramiko',
    'pandas',
    'psutil',
    'pytest',
    'PyYAML',
    'torch',
    'torchvision',
    'torchaudio',
    'ipywidgets',
    'eva-decord',
]}

deps_advanced = {dep: deps_all_dict[dep] for dep in [
    'tables',
    'opt-einsum',
    # 'rolling-quantiles',
    'pulp',
    'spconv',
    'torch-sparse',
    'av',
    'pynwb',
    'sendgrid',
    'pycuda',
    'cuml',
    'cupy',
    'cudf',
    'scanimage-tiff-reader',
    'jupyter',
    'PyWavelets',
    'mat73',
]}

deps_core_latest = {dep: deps_all_latest[dep] for dep in deps_core.keys()}


## Make versions with cv2 headless (for servers)
deps_core_dict_cv2Headless = copy.deepcopy(deps_all_dict)
deps_core_dict_cv2Headless['opencv-contrib-python'] = 'opencv-contrib-python-headless' + deps_core_dict_cv2Headless['opencv-contrib-python'][21:]
deps_core_latest_cv2Headless = copy.deepcopy(deps_all_latest)
deps_core_latest_cv2Headless['opencv-contrib-python'] = 'opencv-contrib-python-headless'


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