#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='one-rep-max',
    version="1.0.2",
    url='https://github.com/slobdell/one_rep_max',
    author="Scott Lobdell",
    author_email="scott.lobdell@gmail.com",
    description=("Basic test with pip install"),
    long_description=("Basic test with pip install"),
    keywords="",
    license="",
    platforms=['linux'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=[],
    extras_require={},
    entry_points = {
        "console_scripts": [
            "sbl-command-name = package.module:main_func_name",
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Topic :: Other/Nonlisted Topic'],
)
