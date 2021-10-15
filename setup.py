#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from os import path
import io
readme = io.open('README.rst').read()
history = io.open('HISTORY.rst').read().replace('.. :changelog:', '')

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

requirements = ['numpy', 'scipy>=0.14.0']

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='astro-limepy',
    version='1.1',
    description='Code to solve lowered isothermal model',
    long_description=long_description + "\n\n" + history,
    long_description_content_type='text/x-rst',
    author='Mark Gieles, Alice Zocchi',
    author_email='m.gieles@surrey.ac.uk, a.zocchi@surrey.ac.uk',
    url='https://github.com/mgieles/limepy',
    packages=[
        'limepy'
    ],
    package_dir={'limepy': 'limepy','sample' : 'sample'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='limepy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
