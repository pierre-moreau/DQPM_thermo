#!/usr/bin/env python

from setuptools import setup

def version():
    with open('./DQPM_thermo/__init__.py', 'r') as f:
        for line in f:
            if line.startswith('__version__ = '):
                return line.split("'")[1]
    raise RuntimeError('unable to determine version')

def requirements():
    req = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            str_line = line.rstrip('\n')
            req.append(str_line)
    return req
    
def long_description():
    with open('README.md') as f:
        return f.read()

setup(
    name='DQPM_thermo',
    version=version(),
    description='To calculate and plot the thermodynamics of the Dynamical QuasiParticle Model (DQPM)',
    long_description=long_description(),
    author='Pierre Moreau',
    author_email='pierre.moreau@duke.edu',
    url='https://github.com/pierre-moreau/DQPM_thermo',
    packages=['DQPM_thermo'],
    package_data={'DQPM_thermo':[]},
    license='MIT',
    install_requires=requirements(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)