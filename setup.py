from setuptools import setup, find_packages

setup(
    name='slic-reveng',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    author='SLIC',
    description='Reverse-engineering subject-level neuroimaging data from meta-analytic summary statistics and interpolating to vertexwise resolution.',
    url='https://github.com/slic-lab/slic-reveng',
)