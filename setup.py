from setuptools import find_packages
from setuptools import setup

required = [
    # please keep alphabetized
    "gym",
    "numpy",
    "pybullet",
]

setup(
    name='metaworld_pybullet',
    version='0.0.0',
    description='Metaworld_Pybullet',
    url='',
    author=(),
    author_email=[],
    license='MIT',
    packages=find_packages(where='metaworld_pybullet'),
    package_dir={'': 'metaworld_pybullet'},
    install_requires=required,
    zip_safe=False
)