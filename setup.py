from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='sadaco',
    version='0.0.2',
    description='Stethoscope Audio Dataset Collections (SADACO)',
    author='KyungChae Lee, Ying Hui Tan',
    author_email='kyungchae.lee@kaist.ac.kr',
    packages=find_packages(where='.'),
    package_dir={'':'.'},
    py_modules=[splitext(basename(path))[0] for path in glob('sadaco/*.py')]
)