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
* setuptools

It is suggested that you check that the required dependencies are installed, although the ```pip```command below should do this automatically for you. You can indeed most conveniently download Concurrent_AP from the official Python Package Index (PyPI) as follows:
* open a terminal window;
* type in the command: ```pip install Concurrent_AP```

The code herewith has been tested on Fedora, OS X and Ubuntu and should work fine on any other member of the Unix-like family of operating systems.

Usage and Command Line Options
------------------------------

See the docstrings associated to each function of the Concurrent_AP module for more information and an understanding of how different tasks are organized and shared among subprocesses.

Usage: ```Concurrent_AP [options] file_name```, where ```file_name``` denotes the path where the data to be processed by Affinity Propagation clustering is held. Each row corresponds to a sample and the features must be tab-separated (*.tsv file). It is also assumed that this file does not display any header.

* ```-c``` or ```--convergence```: specify the number of iterations without change in the number of clusters that signals convergence (defaults to 15);
* ```-d``` or ```--damping```: the damping parameter of Affinity Propagation (defaults to 0.5);
* ```-f``` or ```--file```: option to specify the file name or file handle of the hierarchical data format where the matrices involved in Affinity Propagation clustering will be stored, as part of a group named 'aff_prop_group' at the root of the HDF5 (defaults to a temporary file); the dataset itself won't be accessed there and needn't be stored there either;
* ```-i``` or ```--iterations```: maximum number of message-passing iterations (defaults to 200);
* ```-m``` or ```--multiprocessing```: the number of processes to use;
* ```-p``` or ```--preference```: the preference parameter of Affinity Propagation (if not specified, will be determined as the median of the distribution of pairwise L2 Euclidean distances between samples);
* ```-s``` or ```--similarities```: determine if a similarity matrix has been pre-computed and stored in the HDF5 data structure accessible at the location specified through the command line option ```-f``` or ```--file``` (see above);
* ```-v``` or ```--verbose```: whether to be verbose.

Demo of Concurrent_AP
---------------------

The following few lines illustrate the use of Concurrent_AP on the 'Iris data-set' from the UCI Machine Learning Repository. While the number of samples is here way too small for the benefits of the present multi-tasking implementation and the use of an HDF5 data structure to come fully into play, this data-set has the advantage of being well-known and prone to a quick comparison with scikit-learn's version of Affinity Propagation clustering.

* In a Python interpreter console, enter the following few lines, whose purpose is to create a file containing the Iris data-set that will be later subjected to Affinity Propagation clustering via Concurrent_AP:

```
>>> import numpy as np
>>> from sklearn import datasets

>>> iris = datasets.load_iris()
>>> data = iris.data
>>> with open('./iris_data.tsv', 'w') as f:
        np.savetxt(f, data, fmt = '%.4f', delimiter = '\t')
```

* Open a terminal window.
* Type in: ```Concurrent_AP --preference -5.47 --v iris_data.tsv``` or simply ```Concurrent_AP iris_data.tsv```

The latter will automatically compute a preference parameter from the data-set.

When the rounds of message-passing among data-points have completed, a folder containing a file of cluster labels and a file of cluster centers indices both in tab-separated format is created in your current working directory.

Reference
----------

Brendan J. Frey and Delbert Dueck., "Clustering by Passing Messages Between Data Points".
In: Science, Vol. 315, no. 5814, pp. 972-976. 2007
