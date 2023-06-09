from pathlib import Path

from distutils.core import setup
import copy

dir_parent = Path(__file__).parent

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
deps_all_latest = copy.deepcopy(deps_names)

## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = [deps_all_dict[dep] for dep in [
    'numpy',
    'scipy',
    'matplotlib',
    'numba',
    'scikit-learn',
    'tqdm',
    'h5py',
    'opencv-contrib-python',
    'sparse',
    'natsort',
    'paramiko',
    'pandas',
    'pytest',
    'PyYAML',
    'torch',
    'torchvision',
    'torchaudio',
    'ipywidgets',
    'decord',
]]

deps_advanced = [deps_all_dict[dep] for dep in [
    'tables',
    'opt-einsum',
    'rolling-quantiles',
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
]]


print({
    'deps_all': deps_all,
    'deps_all_latest': deps_all_latest,
    'deps_core': deps_core,
    'deps_advanced': deps_advanced,
})

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
    packages=['bnpm'],
    license='LICENSE',
    description='A library of useful modules for data analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=deps_core,
    extras_require={
        'advanced': deps_advanced,
    },
    url='https://github.com/RichieHakim/basic_neural_processing_modules',
)