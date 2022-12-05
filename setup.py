'''
Wyatt Cupp <wyattcupp@gmail.com>
'''
import re
import os
import codecs
import setuptools

with codecs.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

with codecs.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    DEPENDENCIES = f.read().splitlines()
    if "wheel" not in DEPENDENCIES:
        DEPENDENCIES.append("wheel")

with codecs.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'neural_network', '__init__.py'), encoding='utf-8') as f:
    # version information must always be updated in rule_engine/__init__.py since we want it as an attribute to our package
    VERSION = re.search(
        r'^__version__\s*=\s*([\'"])(?P<version>\d+(\.\d)*)\1$', f.read(), flags=re.MULTILINE).group('version')

if not isinstance(VERSION, str):
    raise RuntimeError(
        "Cannot find version information for neural_network package ...")

DESCRIPTION = 'A MLP neural network implementation completely from scratch, including activation functions, loss functions,\
            forward/backward propagation, gradient descent, custom layers, and batching.'
URL = 'https://github.com/wyattcupp/neural-network-from-scratch'
LICENSE = 'MIT'

setuptools.setup(
    name='neural-network',
    install_requires=DEPENDENCIES,
    tests_require=['pytest'],
    test_suite='tests',
    version=VERSION,
    author='Wyatt Cupp',
    author_email='wyattcupp@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=setuptools.find_packages(include=['neural_network']),
    include_package_data=True
)
