# Concurrent_AP

Overview
--------

A scalable and concurrent programming implementation of Affinity Propagation clustering.

Affinity Propagation is a clustering algorithm based on passing messages between data-points.

Storing and updating matrices of 'affinities', 'responsibilities' and 'similarities' between samples can be memory-intensive.
We address this issue through the use of an HDF5 data structure, allowing Affinity Propagation clustering of arbitrary large data-sets, where other Python implementations would return a MemoryError on most machines.

We also significantly speed up the computations by splitting them up across subprocesses, thereby taking full advantage of the resources of multi-core processors and bypassing the Global Interpreter Lock of the standard Python interpreter, CPython.

Installation and Requirements
-----------------------------

Concurrent_AP requires Python 2.7 along with the following packages and a few modules from the Standard Python Library:
* NumPy >= 1.9
* psutil
* PyTables
* scikit-learn

Upon checking that the required dependencies are installed, you can upload Concurrent_AP from the official Python Package Index (PyPI) as follows:
* open a terminal window;
* type in the command: ''pip install Concurrent_AP''

Usage and Exemple
-----------------

See the docstrings associated to each function of the Concurrent_AP module for morre information and an understanding of how different tasks are organized and shared across subprocesses.

The following few lines illustrate the use of Concurrent_AP on the 'Iris data-set' from the UCI Machine Learning Repository. While the number of samples is here way too small for the benefits of the present multi-tasking implementation and the use of an HDF5 data structure to come into play, this data-set comes with the advantage of being well-known and prone to a quick comparison with scikit-learn's version of Affinity Propagation clustering.

* In a Python interpreter console, enter the following few lines, whose purpose is to create a file containing the Iris data-set to be later subjected to Affinity Propagation clustering via Concurrent_AP:

'''
>>> import numpy as np
>>> from sklearn import datasets

>>> iris = datasets.load_iris()
>>> data = iris.data
>>> with open('./iris_data.txt', 'w') as f:
    np.loadtxt(f, data, fmt = '%.4f')
'''

* Open a terminal window.
* Type in: ''./Concurrent_AP --preference 5.47 --verbose iris_data.txt'' or simply ''./Concurrent_AP iris_data.txt''

The latter will automatically compute a preference parameter from the data-set.

When the rounds of message-passing among data-points have completed, a folder containing a file of cluster labels and a file of cluster centers indices both in tab-separated format is created in your current working directory.

Command Line Options
--------------------



* -c or --convergence: specify the number of iterations without change in the number of clusters that signals convergence (defaults to 15);
* -d or --damping: the damping parameter of Affinity Propagation (defaults to 0.5);
* -f or --file: option to specify the file name or file handle of the hierarchical data format where the matrices involved in Affinity Propagation clustering will be stored (defaults to a temporary file);
* -i or --iterations: maximum number of message-passing iterations (defaults to 200);
* -m or --multiprocessing: the number of processes to use;
* -p or --preference: the preference parameter of Affinity Propagation (if not specified, will be determined as the median of the distribution of pairwise L2 Euclidean distances between samples);
* -s or --similarities: determine if a similarity matrix has been pre-computed and stored in the HDF5 data structure accessible at the location specified through the command line option -f or --file (see above);
* -v or --verbose: whether to be verbose.

References
----------

Brendan J. Frey and Delbert Dueck. "Clustering by Passing Messages between Data Points", Science Feb. 2007
