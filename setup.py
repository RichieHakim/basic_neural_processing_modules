## setup.py file for roicat

from distutils.core import setup

setup(
    name='bnpm',
    version='0.1.0',
    author='Richard Hakim',
    keywords=['data analysis', 'machine learning', 'neuroscience'],
    packages=['bnpm'],
    license='LICENSE',
    description='A library of useful modules for data analysis.',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/RichieHakim/basic_neural_processing_modules',
)