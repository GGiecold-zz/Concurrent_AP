#!/usr/bin/env python


# Concurrent_AP/setup.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Setup script for Concurrent_AP, a scalable and concurrent programming
implementation of Affinity Propagation clustering. 

Affinity Propagation is a clustering algorithm based on passing 
messages between data-points. 

Storing and updating matrices of 'affinities', 'responsibilities' 
and 'similarities' between samples can be memory-intensive. 
We address this issue through the use of an HDF5 data structure, 
allowing Affinity Propagation clustering of arbitrarily large data-sets, 
where other Python implementations would return a MemoryError 
on most machines.

We also significantly speed up the computations by splitting them up 
across subprocesses, thereby taking full advantage of the resources 
of multi-core processors and bypassing the Global Interpreter Lock 
of the standard Python interpreter, CPython.

Reference
---------
Brendan J. Frey and Delbert Dueck., "Clustering by Passing Messages Between Data Points". 
In: Science, Vol. 315, no. 5814, pp. 972-976. 2007
"""


from codecs import open
from os import path
from sys import version
from setuptools import setup, find_packages

    
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()
    

setup(name = 'Concurrent_AP',
      version = '1.4',
      
      description = 'Scalable and parallel programming implementation of Affinity Propagation clustering',
      long_description = long_description,
                    
      url = 'https://github.com/GGiecold/Concurrent_AP',
      download_url = 'https://github.com/GGiecold/Concurrent_AP',
      
      author = 'Gregory Giecold',
      author_email = 'g.giecold@gmail.com',
      maintainer = 'Gregory Giecold',
      maintainer_email = 'ggiecold@jimmy.harvard.edu',
      
      license = 'MIT License',
      
      packages = find_packages(),
      
      py_modules = ['Concurrent_AP'],
      platforms = ('Any',),
      install_requires = ['numpy>=1.9.0', 'psutil', 'sklearn', 'setuptools', 'tables'],
                          
      classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',          
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Mathematics', ],
                   
      keywords = 'parallel multiprocessing machine-learning concurrent clustering',
      
      entry_points = {
          'console_scripts': ['Concurrent_AP = Concurrent_AP:main'],
          }    
)

