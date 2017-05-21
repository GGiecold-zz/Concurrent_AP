#!/usr/bin/env python


# Concurrent_AP/Concurrent_AP.py

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Concurrent_AP is a scalable and concurrent programming implementation 
of Affinity Propagation clustering. 

Affinity Propagation is a clustering algorithm based on passing messages 
between data-points. 

Storing and updating matrices of 'affinities', 'responsibilities' and 
'similarities' between samples can be memory-intensive. 
We address this issue through the use of an HDF5 data structure, 
allowing Affinity Propagation clustering of arbitrarily large data-sets, 
where other Python implementations would return a MemoryError on most machines.

We also significantly speed up the computations by splitting them up 
across subprocesses, thereby taking full advantage of the resources 
of multi-core processors and bypassing the Global Interpreter Lock 
of the standard Python interpreter, CPython.

Reference
---------
Brendan J. Frey and Delbert Dueck., "Clustering by Passing Messages Between Data Points". 
In: Science, Vol. 315, no. 5814, pp. 972-976. 2007
"""


from abc import ABCMeta, abstractmethod
from contextlib import closing
from ctypes import c_double, c_int
import gc
import multiprocessing
import numpy as np
import optparse
import os
import psutil
from sklearn.metrics.pairwise import euclidean_distances
import sys
import tables
from tempfile import NamedTemporaryFile
import time
import warnings

np.seterr(invalid = 'ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)


__all__ = []


def memory():
    """Determine memory specifications of the machine.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.
    """

    mem_info = dict()

    for k, v in psutil.virtual_memory().__dict__.items():
           mem_info[k] = int(v)
           
    return mem_info


def get_chunk_size(N, n):
    """Given a two-dimensional array with a dimension of size 'N', 
        determine the number of rows or columns that can fit into memory.

    Parameters
    ----------
    N : int
        The size of one of the dimensions of a two-dimensional array.  

    n : int
        The number of arrays of size 'N' times 'chunk_size' that can fit in memory.

    Returns
    -------
    chunk_size : int
        The size of the dimension orthogonal to the one of size 'N'. 
    """

    mem_free = memory()['free']
    if mem_free > 60000000:
        chunk_size = int(((mem_free - 10000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 40000000:
        chunk_size = int(((mem_free - 7000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 14000000:
        chunk_size = int(((mem_free - 2000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 8000000:
        chunk_size = int(((mem_free - 1400000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 2000000:
        chunk_size = int(((mem_free - 900000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 1000000:
        chunk_size = int(((mem_free - 400000) * 1000) / (4 * n * N))
        return chunk_size
    else:
        print("\nERROR: Concurrent_AP: get_chunk_size: this machine does not "
              "have enough free memory.\n")
        sys.exit(1)


def chunk_generator(N, n):
    """Returns a generator of slice objects.
    
    Parameters
    ----------
    N : int
        The size of one of the dimensions of a two-dimensional array. 
    
    n : int
        The number of arrays of shape ('N', 'get_chunk_size(N, n)') that fit into
        memory.
    
    Returns
    -------
    Slice objects of the type 'slice(start, stop)' are generated, representing
    the set of indices specified by 'range(start, stop)'. 
    """

    chunk_size = get_chunk_size(N, n)
    
    for start in range(0, N, chunk_size):
        yield slice(start, min(start + chunk_size, N))
    
    
def parse_options():
    """Specify the command line options to parse.
    
    Returns
    -------
    opts : optparse.Values instance
        Contains the option values in its 'dict' member variable.
    
    args[0] : string or file-handler
        The name of the file storing the data-set submitted
        for Affinity Propagation clustering.
    """

    parser = optparse.OptionParser(
                        usage = "Usage: %prog [options] file_name\n\n"
                        "file_name denotes the path where the data to be "
                        "processed by affinity propagation clustering is stored"
                        )
    parser.add_option('-m', '--multiprocessing', dest = 'count', 
                      default = multiprocessing.cpu_count(), type = 'int', 
                      help = ("The number of processes to use (1..20) " 
                              "[default %default]"))
    parser.add_option('-f', '--file', dest = 'hdf5_file', default = None,
                     type = 'str', 
                     help = ("File name or file handle of the HDF5 "
                             "data structure holding the matrices involved in "
                             "affinity propagation clustering "
                             "[default %default]"))
    parser.add_option('-s', '--similarities', dest = 'similarities', 
                      default = False, action = 'store_true',
                      help = ("Specifies if a matrix of similarities "
                              "has already been computed; only makes sense "
                              "with -f or --file in effect [default %default]"))
    parser.add_option('-i', '--iterations', dest = 'max_iter', 
                      default = 200, type = 'int', 
                      help = ("The maximum number of message passing "   
                              "iterations undergone before affinity "
                              "propagation returns, having reached "
                              "convergence or not [default %default]"))
    parser.add_option('-c', '--convergence', dest = 'convergence_iter', 
                      default = 15, type = 'int', 
                      help = ("Specifies the number of consecutive "
                              "iterations without change in the number "
                              "of clusters that signals convergence "
                              "[default %default]") )
    parser.add_option('-p', '--preference', dest = 'preference', 
                      default = None, type = 'float', 
                      help = ("The preference parameter of affinity "
                              "propagation [default %default]"))
    parser.add_option('-d', '--damping', dest = 'damping',
                      default = 0.5, type = 'float',
                      help = ("The damping parameter of affinity " 
                              "propagation; must be within 0.5 and 1.0 "
                              "[default %default]"))
    parser.add_option('-v', '--verbose', dest = 'verbose', 
                      default = False, action = 'store_true', 
                      help = ("Turns on the display of messaging regarding " 
                              "the status of the various stages of affinity "
                              "propagation clustering currently ongoing "
                              "on the user-specified data-set "
                              "[default %default]"))
                    
    opts, args = parser.parse_args()
    
    if len(args) == 0:
        parser.error('A data file must be specified')
    
    if opts.similarities and (opts.hdf5_file is None):
        parser.error("Option -s is conditional on -f")
    
    if not (1 <= opts.count <= 20):
        parser.error("The number of processes must range "
                     "from 1 to 20, inclusive")
                     
    if opts.max_iter <= 0:
        parser.error("The number of iterations must be "
                     "a non-negative integer")
                                      
    if opts.convergence_iter >= opts.max_iter:
        parser.error("The number of iterations signalling convergence "
                     "cannot exceed the maximum number of iterations possibly "
                     "required")
                     
    if not (0.5 <= opts.damping <= 1.0):
        parser.error("The damping parameter is restricted to values "
                     "between 0.5 and 1.0")
        
    return opts, args[0]
    
    
def check_HDF5_arrays(hdf5_file, N, convergence_iter):
    """Check that the HDF5 data structure of file handle 'hdf5_file' 
        has all the required nodes organizing the various two-dimensional 
        arrays required for Affinity Propagation clustering 
        ('Responsibility' matrix, 'Availability', etc.).
    
    Parameters
    ----------
    hdf5_file : string or file handle
        Name of the Hierarchical Data Format under consideration.
        
    N : int
        The number of samples in the data-set that will undergo Affinity Propagation
        clustering.
    
    convergence_iter : int
        Number of iterations with no change in the number of estimated clusters 
        that stops the convergence.
    """
    
    Worker.hdf5_lock.acquire()

    with tables.open_file(hdf5_file, 'r+') as fileh:
        if not hasattr(fileh.root, 'aff_prop_group'):
            fileh.create_group(fileh.root, "aff_prop_group")

        atom = tables.Float32Atom()
        filters = None
        #filters = tables.Filters(5, 'blosc')
            
        for feature in ('availabilities', 'responsibilities',
                            'similarities', 'temporaries'):
            if not hasattr(fileh.root.aff_prop_group, feature):
                fileh.create_carray(fileh.root.aff_prop_group, feature, 
                         atom, (N, N), "Matrix of {0} for affinity "
                         "propagation clustering".format(feature), 
                         filters = filters)

        if not hasattr(fileh.root.aff_prop_group, 'parallel_updates'):
            fileh.create_carray(fileh.root.aff_prop_group,
                     'parallel_updates', atom, (N, convergence_iter), 
                     "Matrix of parallel updates for affinity propagation "
                     "clustering", filters = filters)
                     
    Worker.hdf5_lock.release()

        
class Worker(multiprocessing.Process, metaclass=ABCMeta):
    """Abstract Base Class whose methods are meant to be overriden 
        by the various classes of processes designed to handle 
        the various stages of Affinity Propagation clustering.
    """
    
    hdf5_lock = multiprocessing.Lock()
    
    @abstractmethod
    def __init__(self, hdf5_file, path, slice_queue):
        multiprocessing.Process.__init__(self)
        self.hdf5_file = hdf5_file
        self.path = path
        self.slice_queue = slice_queue
        
    def run(self):
        while True:
            try:
                slc = self.slice_queue.get()
                self.process(slc)            
            finally:
                self.slice_queue.task_done()
                
    @abstractmethod
    def process(self, slc):
        raise NotImplementedError() 


class Similarities_worker(Worker):
    """Class of worker processes handling the computation of 
        a similarities matrix of pairwise distances between samples.
    """
        
    def __init__(self, hdf5_file, path, array, slice_queue):
        super(self.__class__, self).__init__(hdf5_file, path, slice_queue)
        self.array = array
        
    def process(self, rows_slice):
        tmp = self.array[rows_slice, ...]
        result = - euclidean_distances(tmp, self.array, squared = True)

        with Worker.hdf5_lock:            
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                hdf5_array = fileh.get_node(self.path)
                hdf5_array[rows_slice, ...] = result
                
        del tmp


class Fluctuations_worker(Worker):
    """Class of worker processes adding small random fluctuations 
        to the array specified by the node accessed via 'path' in 'hdf5_file'.  
    """    
        
    def __init__(self, hdf5_file, path, random_state, N, slice_queue):
        super(self.__class__, self).__init__(hdf5_file, path, slice_queue)
        self.random_state = random_state
        self.N = N
        
    def process(self, rows_slice):
        with Worker.hdf5_lock:
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                hdf5_array = fileh.get_node(self.path)
                X = hdf5_array[rows_slice, ...]
        
        eensy = np.finfo(np.float32).eps
        weensy =  np.finfo(np.float32).tiny * 100
        tmp = self.random_state.randn(rows_slice.stop - rows_slice.start, self.N)
        X += (eensy * X + weensy) * tmp
        
        with Worker.hdf5_lock:
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                hdf5_array = fileh.get_node(self.path)
                hdf5_array[rows_slice, ...] = X
                
        del X
            
            
class Responsibilities_worker(Worker):
    """Class of worker processes that are tasked with computing 
        and updating the responsibility matrix.
    """

    def __init__(self, hdf5_file, path, N, damping, slice_queue):
        super(self.__class__, self).__init__(hdf5_file, path, slice_queue)
        self.N = N
        self.damping = damping
        
    def process(self, rows_slice):
        Worker.hdf5_lock.acquire()
        
        with tables.open_file(self.hdf5_file, 'r+') as fileh:
            A = fileh.get_node(self.path + '/availabilities')
            S = fileh.get_node(self.path + '/similarities')
            T = fileh.get_node(self.path + '/temporaries')
            
            s = S[rows_slice, ...]
            a = A[rows_slice, ...]
        
        Worker.hdf5_lock.release()
        
        ind = np.arange(0, rows_slice.stop - rows_slice.start)    
        
        tmp = a + s    
        I = tmp.argmax(axis = 1)
        Y = tmp[ind, I]
        tmp[ind, I] = - np.inf
        Y_2 = tmp.max(axis = 1)
        # tmp = R_new
        np.subtract(s, Y[:, None], tmp)
        tmp[ind, I] = s[ind, I] - Y_2
        
        with Worker.hdf5_lock:
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                R = fileh.get_node(self.path + '/responsibilities')
                r = R[rows_slice, ...]
        
        # damping
        r = r * self.damping + tmp * (1 - self.damping)
        # tmp = R_p
        tmp = np.where(r >= 0, r, 0)
        tmp[ind, rows_slice.start + ind] = r[ind, rows_slice.start + ind]
        
        Worker.hdf5_lock.acquire()
        
        with tables.open_file(self.hdf5_file, 'r+') as fileh:
            R = fileh.get_node(self.path + '/responsibilities')
            T = fileh.get_node(self.path + '/temporaries')
        
            R[rows_slice, ...] = r 
            T[rows_slice, ...] = tmp
        
        Worker.hdf5_lock.release()
        
        del a, r, s, tmp
        

class Rows_worker(Worker):
    """The processes instantiated from this class compute the sums 
        of row entries in an array accessed at node 'path' from the 
        hierarchidal data format at 'hdf5_file'. Those sums are stored 
        in the shared multiprocessing.Array data structure 'g_rows_sum'.
    """

    def __init__(self, hdf5_file, path, N, slice_queue, g_rows_sum):
        super(self.__class__, self).__init__(hdf5_file, path, slice_queue)
        self.N = N
        self.g_rows_sum = g_rows_sum
        
    def process(self, rows_slice):
        get_sum(self.hdf5_file, self.path, self.g_rows_sum, 
                out_lock, rows_slice)        
     
            
def get_sum(hdf5_file, path, array_out, out_lock, rows_slice):
    """Access an array at node 'path' of the 'hdf5_file', compute the sums 
        along a slice of rows specified by 'rows_slice' and add the resulting 
        vector to 'array_out'.
    
    Parameters
    ----------
    hdf5_file : string or file handle
        The location of the HDF5 data structure containing the matrices of availabitilites,
        responsibilities and similarities among others.
        
    path : string 
        Specify the node where the matrix whose row-sums are to be computed is located
        within the given hierarchical data format.
    
    array_out : multiprocessing.Array object
        This ctypes array is allocated from shared memory and used by various
        processes to store the outcome of their computations.
    
    out_lock : multiprocessing.Lock object
        Synchronize access to the values stored in 'array_out'.
        
    rows_slice : slice object
        Specifies a range of rows indices.
    """
    
    Worker.hdf5_lock.acquire()
    
    with tables.open_file(hdf5_file, 'r+') as fileh:
        hdf5_array = fileh.get_node(path)
        tmp = hdf5_array[rows_slice, ...]
        
    Worker.hdf5_lock.release()
    
    szum = np.sum(tmp, axis = 0)
    with out_lock: 
        array_out += szum
        
    del tmp
    
                
class Availabilities_worker(Worker):
    """Class of processes working on the computation and update of the 
        availability matrix for Affinity Propagation Clustering.
    """

    def __init__(self, hdf5_file, path, N, damping, slice_queue, rows_sum):
        super(self.__class__, self).__init__(hdf5_file, path, slice_queue)
        self.N = N
        self.damping = damping
        self.rows_sum = rows_sum
        
    def process(self, rows_slice):
                    
        with Worker.hdf5_lock:
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                T = fileh.get_node(self.path + '/temporaries')
                tmp = T[rows_slice, ...]
            
        ind = np.arange(0, rows_slice.stop - rows_slice.start)    
        
        # tmp = - A_new    
        tmp -= self.rows_sum
        diag_A = tmp[ind, rows_slice.start + ind].copy()
        np.clip(tmp, 0, np.inf, tmp)
        tmp[ind, rows_slice.start + ind] = diag_A
        
        Worker.hdf5_lock.acquire()
        
        with tables.open_file(self.hdf5_file, 'r+') as fileh:
            A = fileh.get_node(self.path + '/availabilities')
            a = A[rows_slice, ...]
            
        Worker.hdf5_lock.release()
            
        # yet more damping
        a = a * self.damping - tmp * (1 - self.damping)
        
        with Worker.hdf5_lock:
            with tables.open_file(self.hdf5_file, 'r+') as fileh:
                A = fileh.get_node(self.path + '/availabilities')
                T = fileh.get_node(self.path + '/temporaries')
                
                A[rows_slice, ...] = a
                T[rows_slice, ...] = tmp
                
        del a, tmp


def terminate_processes(pid_list):
    """Terminate a list of processes by sending to each of them a SIGTERM signal, 
        pre-emptively checking if its PID might have been reused.
    
    Parameters
    ----------
    pid_list : list
        A list of process identifiers identifying active processes.
    """

    for proc in psutil.process_iter():
        if proc.pid in pid_list:
            proc.terminate()
             
                  
def compute_similarities(hdf5_file, data, N_processes):
    """Compute a matrix of pairwise L2 Euclidean distances among samples from 'data'.
        This computation is to be done in parallel by 'N_processes' distinct processes. 
        Those processes (which are instances of the class 'Similarities_worker') 
        are prevented from simultaneously accessing the HDF5 data structure 
        at 'hdf5_file' through the use of a multiprocessing.Lock object.
    """

    slice_queue = multiprocessing.JoinableQueue()
    
    pid_list = []
    for i in range(N_processes):
        worker = Similarities_worker(hdf5_file, '/aff_prop_group/similarities',
                                     data, slice_queue)
        worker.daemon = True
        worker.start()
        pid_list.append(worker.pid)
    
    for rows_slice in chunk_generator(data.shape[0], 2 * N_processes):
        slice_queue.put(rows_slice)
        
    slice_queue.join()    
    slice_queue.close()
    
    terminate_processes(pid_list)
    gc.collect()
    

def add_preference(hdf5_file, preference):
    """Assign the value 'preference' to the diagonal entries
        of the matrix of similarities stored in the HDF5 data structure 
        at 'hdf5_file'.
    """

    Worker.hdf5_lock.acquire()
    
    with tables.open_file(hdf5_file, 'r+') as fileh:
        S = fileh.root.aff_prop_group.similarities
        diag_ind = np.diag_indices(S.nrows)
        S[diag_ind] = preference
        
    Worker.hdf5_lock.release()
            

def add_fluctuations(hdf5_file, N_columns, N_processes):
    """This procedure organizes the addition of small fluctuations on top of 
        a matrix of similarities at 'hdf5_file' across 'N_processes' 
        different processes. Each of those processes is an instance of the 
        class 'Fluctuations_Worker' defined elsewhere in this module.
    """

    random_state = np.random.RandomState(0)
        
    slice_queue = multiprocessing.JoinableQueue()
        
    pid_list = []
    for i in range(N_processes):
        worker = Fluctuations_worker(hdf5_file,
                   '/aff_prop_group/similarities', random_state, 
                   N_columns, slice_queue)
        worker.daemon = True
        worker.start()
        pid_list.append(worker.pid)
            
    for rows_slice in chunk_generator(N_columns, 4 * N_processes):
        slice_queue.put(rows_slice)
        
    slice_queue.join()
    slice_queue.close()
    
    terminate_processes(pid_list)
    gc.collect()
    
    
def compute_responsibilities(hdf5_file, N_columns, damping, N_processes):
    """Organize the computation and update of the responsibility matrix
        for Affinity Propagation clustering with 'damping' as the eponymous 
        damping parameter. Each of the processes concurrently involved in this task 
        is an instance of the class 'Responsibilities_worker' defined above.
    """

    slice_queue = multiprocessing.JoinableQueue()
    
    pid_list = []
    for i in range(N_processes):
        worker = Responsibilities_worker(hdf5_file, '/aff_prop_group',
                   N_columns, damping, slice_queue)
        worker.daemon = True
        worker.start()
        pid_list.append(worker.pid)
        
    for rows_slice in chunk_generator(N_columns, 8 * N_processes):
        slice_queue.put(rows_slice)
        
    slice_queue.join()
    slice_queue.close()
    
    terminate_processes(pid_list)
    
    
def rows_sum_init(hdf5_file, path, out_lock, *numpy_args):
    """Create global variables sharing the same object as the one pointed by
        'hdf5_file', 'path' and 'out_lock'.
        Also Create a NumPy array copy of a multiprocessing.Array ctypes array 
        specified by '*numpy_args'.
    """

    global g_hdf5_file, g_path, g_out, g_out_lock
        
    g_hdf5_file, g_path, g_out_lock = hdf5_file, path, out_lock
    g_out = to_numpy_array(*numpy_args)
        

def multiprocessing_get_sum(columns_slice):

    get_sum(g_hdf5_file, g_path, g_out, g_out_lock, columns_slice)    
     
    
def to_numpy_array(multiprocessing_array, shape, dtype):
    """Convert a share multiprocessing array to a numpy array.
        No data copying involved.
    """

    return np.frombuffer(multiprocessing_array.get_obj(),
                         dtype = dtype).reshape(shape)


def compute_rows_sum(hdf5_file, path, N_columns, N_processes, method = 'Process'):
    """Parallel computation of the sums across the rows of two-dimensional array
        accessible at the node specified by 'path' in the 'hdf5_file' 
        hierarchical data format. 
    """                 

    assert isinstance(method, str), "parameter 'method' must consist in a string of characters"
    assert method in ('Ordinary', 'Pool'), "parameter 'method' must be set to either of 'Ordinary' or 'Pool'"
    
    if method == 'Ordinary':
        rows_sum = np.zeros(N_columns, dtype = float)
        
        chunk_size = get_chunk_size(N_columns, 2)
        with Worker.hdf5_lock:
            with tables.open_file(hdf5_file, 'r+') as fileh:
                hdf5_array = fileh.get_node(path)
                
                N_rows = hdf5_array.nrows
                assert N_columns == N_rows
                
                for i in range(0, N_columns, chunk_size):
                    slc = slice(i, min(i+chunk_size, N_columns))
                    tmp = hdf5_array[:, slc]
                    rows_sum[slc] = tmp[:].sum(axis = 0)
                        
    else:
        rows_sum_array = multiprocessing.Array(c_double, N_columns, lock = True)
    
        chunk_size = get_chunk_size(N_columns, 2 * N_processes)
        numpy_args = rows_sum_array, N_columns, np.float64
    
        with closing(multiprocessing.Pool(N_processes, 
                     initializer = rows_sum_init, 
                     initargs = (hdf5_file, path, rows_sum_array.get_lock()) +
                     numpy_args)) as pool:
            pool.map_async(multiprocessing_get_sum, 
                  chunk_generator(N_columns, 2 * N_processes), chunk_size)
        
        pool.close()
        pool.join()
    
        rows_sum = to_numpy_array(*numpy_args)
    
    gc.collect()
    
    return rows_sum
        

def compute_availabilities(hdf5_file, N_columns, damping, N_processes, rows_sum):
    """Coordinates the computation and update of the availability matrix
        for Affinity Propagation clustering. 
    
    Parameters
    ----------
    hdf5_file : string or file handle
        Specify access to the hierarchical data format used throughout all the iterations
        of message-passing between data-points involved in Affinity Propagation clustering.
    
    N_columns : int
        The number of samples in the data-set subjected to Affinity Propagation clustering.
    
    damping : float
        The damping parameter of Affinity Propagation clustering, typically set to 0.5.
    
    N_processes : int
        The number of subprocesses involved in the parallel computation and update of the
        matrix of availabitilies.
    
    rows_sum : array of shape (N_columns,)
        A vector containing, for each column entry of the similarities matrix, the sum
        of its rows entries. 
    """
    
    slice_queue = multiprocessing.JoinableQueue()
    
    pid_list = []
    for i in range(N_processes):
        worker = Availabilities_worker(hdf5_file, '/aff_prop_group',
                   N_columns, damping, slice_queue, rows_sum)
        worker.daemon = True
        worker.start()
        pid_list.append(worker.pid)
        
    for rows_slice in chunk_generator(N_columns, 8 * N_processes):
        slice_queue.put(rows_slice)
        
    slice_queue.join()
    slice_queue.close()
    
    terminate_processes(pid_list)
    gc.collect()
    
    
def check_convergence(hdf5_file, iteration, convergence_iter, max_iter):
    """If the estimated number of clusters has not changed for 'convergence_iter'
        consecutive iterations in a total of 'max_iter' rounds of message-passing, 
        the procedure herewith returns 'True'.
        Otherwise, returns 'False'.
        Parameter 'iteration' identifies the run of message-passing 
        that has just completed.
    """

    Worker.hdf5_lock.acquire()
    
    with tables.open_file(hdf5_file, 'r+') as fileh:
        A = fileh.root.aff_prop_group.availabilities
        R = fileh.root.aff_prop_group.responsibilities
        P = fileh.root.aff_prop_group.parallel_updates
        
        N = A.nrows
        diag_ind = np.diag_indices(N)
        
        E = (A[diag_ind] + R[diag_ind]) > 0
        P[:, iteration % convergence_iter] = E
        
        e_mat = P[:]
        K = E.sum(axis = 0)
        
    Worker.hdf5_lock.release()
        
    if iteration >= convergence_iter:
        se = e_mat.sum(axis = 1)
        unconverged = (np.sum((se == convergence_iter) + (se == 0)) != N)
        if (not unconverged and (K > 0)) or (iteration == max_iter):
                return True
                
    return False
    

def cluster_labels_init(hdf5_file, I, c_array_lock, *numpy_args):

    global g_hdf5_file, g_I, g_c_array_lock, g_c
    
    g_hdf5_file, g_I, g_c_array_lock = hdf5_file, I, c_array_lock
    g_c = to_numpy_array(*numpy_args)


def cluster_labels_init_B(hdf5_file, I, ii, iix, s_reduced_array_lock,
                         *numpy_args):

    global g_hdf5_file, g_I, g_ii, g_iix, g_s_reduced_array_lock, g_s_reduced
    
    g_hdf5_file, g_I, g_ii, g_iix = hdf5_file, I, ii, iix
    g_s_reduced_array_lock = s_reduced_array_lock
    g_s_reduced = to_numpy_array(*numpy_args)
    

def multiprocessing_cluster_labels_A(rows_slice):

    cluster_labels_A(g_hdf5_file, g_c, g_c_array_lock, g_I, rows_slice)


def multiprocessing_cluster_labels_B(rows_slice):

    cluster_labels_B(g_hdf5_file, g_s_reduced, g_s_reduced_array_lock, 
                     g_I, g_ii, g_iix, rows_slice)


def multiprocessing_cluster_labels_C(rows_slice):

    cluster_labels_C(g_hdf5_file, g_c, g_c_array_lock, g_I, rows_slice)


def cluster_labels_A(hdf5_file, c, lock, I, rows_slice):
    """One of the task to be performed by a pool of subprocesses, as the first
        step in identifying the cluster labels and indices of the cluster centers
        for Affinity Propagation clustering.
    """

    with Worker.hdf5_lock:
        with tables.open_file(hdf5_file, 'r+') as fileh:
            S = fileh.root.aff_prop_group.similarities
            s = S[rows_slice, ...]
        
    s = np.argmax(s[:, I], axis = 1)
          
    with lock:        
        c[rows_slice] = s[:]
        
    del s


def cluster_labels_B(hdf5_file, s_reduced, lock, I, ii, iix, rows_slice):
    """Second task to be performed by a pool of subprocesses before
        the cluster labels and cluster center indices can be identified.
    """

    with Worker.hdf5_lock:
        with tables.open_file(hdf5_file, 'r+') as fileh:
            S = fileh.root.aff_prop_group.similarities
            s = S[rows_slice, ...]
           
    s = s[:, ii]
    s = s[iix[rows_slice]]
    
    with lock:                
        s_reduced += s[:].sum(axis = 0)
        
    del s


def cluster_labels_C(hdf5_file, c, lock, I, rows_slice):
    """Third and final task to be executed by a pool of subprocesses, as part of the
        goal of finding the cluster to which each data-point has been assigned by 
        Affinity Propagation clustering on a given data-set.
    """

    with Worker.hdf5_lock:
        with tables.open_file(hdf5_file, 'r+') as fileh:
            S = fileh.root.aff_prop_group.similarities
            s = S[rows_slice, ...]
            
    s = s[:, I]
    
    with lock:
        c[rows_slice] = np.argmax(s[:], axis = 1)
        
    del s
        

def get_cluster_labels(hdf5_file, N_processes):
    """
    Returns
    -------
    cluster_centers_indices : array of shape (n_clusters,)
        Indices of cluster centers
        
    labels : array of shape (n_samples,)
        Specify the label of the cluster to which each point has been assigned.
    """

    with Worker.hdf5_lock:
        with tables.open_file(hdf5_file, 'r+') as fileh:
            A = fileh.root.aff_prop_group.availabilities
            R = fileh.root.aff_prop_group.responsibilities
    
            N = A.nrows
            diag_ind = np.diag_indices(N)
            
            a = A[diag_ind]
            r = R[diag_ind]
    
    I = np.where(np.add(a[:], r[:]) > 0)[0]
    K = I.size
    
    if K == 0:
        labels = np.empty((N, 1))
        labels.fill(np.nan)
        cluster_centers_indices = None
        
    else:
        c_array = multiprocessing.Array(c_int, N, lock = True)
    
        chunk_size = get_chunk_size(N, 3 * N_processes)
        numpy_args = c_array, N, np.int32
    
        with closing(multiprocessing.Pool(N_processes, 
                     initializer = cluster_labels_init, 
                     initargs = (hdf5_file, I, c_array.get_lock()) 
                               + numpy_args)) as pool:
            pool.map_async(multiprocessing_cluster_labels_A, 
                  chunk_generator(N, 3 * N_processes), chunk_size)
        
        pool.close()
        pool.join()
        
        gc.collect()
        
        c = to_numpy_array(*numpy_args)
        c[I] = np.arange(K)
        # determine the exemplars of clusters, applying some 
        # cosmetics to our results before returning them
        for k in range(K):
            ii = np.where(c == k)[0]
            
            iix = np.full(N, False, dtype = bool)
            iix[ii] = True
            
            s_reduced_array = multiprocessing.Array(c_double, ii.size, 
                                                    lock = True)
        
            chunk_size = get_chunk_size(N, 3 * N_processes)
            numpy_args = s_reduced_array, ii.size, np.float64
            
            with closing(multiprocessing.Pool(N_processes,
                         initializer = cluster_labels_init_B,
                         initargs = (hdf5_file, I, ii, iix, 
                         s_reduced_array.get_lock())
                         + numpy_args)) as pool:
                pool.map_async(multiprocessing_cluster_labels_B,
                      chunk_generator(N, 3 * N_processes), chunk_size)
                               
            pool.close()
            pool.join()
            
            s_reduced = to_numpy_array(*numpy_args)
            
            j = np.argmax(s_reduced)
            I[k] = ii[j]
            
            gc.collect()
            
        c_array = multiprocessing.Array(c_int, N, lock = True)
        
        chunk_size = get_chunk_size(N, 3 * N_processes)
        numpy_args = c_array, N, np.int32
    
        with closing(multiprocessing.Pool(N_processes, 
                     initializer = cluster_labels_init, 
                     initargs = (hdf5_file, I, c_array.get_lock()) 
                               + numpy_args)) as pool:
            pool.map_async(multiprocessing_cluster_labels_C, 
                  chunk_generator(N, 3 * N_processes), chunk_size)
        
        pool.close()
        pool.join()
        
        c = to_numpy_array(*numpy_args)
        c[I] = np.arange(K)
        labels = I[c]
        
        gc.collect()

        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
        
    return cluster_centers_indices, labels


def output_clusters(labels, cluster_centers_indices):
    """Write in tab-separated files the vectors of cluster identities and
        of indices of cluster centers.
    """
            
    here = os.getcwd()
    try:
        output_directory = os.path.join(here, 'concurrent_AP_output')
        os.makedirs(output_directory)
    except OSError:
        if not os.path.isdir(output_directory):
            print("ERROR: concurrent_AP: output_clusters: cannot create a directory "
                  "for storage of the results of Affinity Propagation clustering "
                  "in your current working directory")
            sys.exit(1)
            
    if any(np.isnan(labels)):
        fmt = '%.1f'
    else:
        fmt = '%d'
                
    with open(os.path.join(output_directory, 'labels.tsv'), 'w') as fh:
        np.savetxt(fh, labels, fmt = fmt, delimiter = '\t')
        
    if cluster_centers_indices is not None:    
        with open(os.path.join(output_directory, 'cluster_centers_indices.tsv'), 'w') as fh:
            np.savetxt(fh, cluster_centers_indices, fmt = '%.1f', 
                       delimiter = '\t')


def set_preference(data, chunk_size):
    """Return the median of the distribution of pairwise L2 Euclidean distances 
        between samples (the rows of 'data') as the default preference parameter
        for Affinity Propagation clustering.

    Parameters
    ----------
    data : array of shape (N_samples, N_features)
        The data-set submitted for Affinity Propagation clustering.
        
    chunk_size : int
        The size of random subsamples from the data-set whose similarity
        matrix is computed. The resulting median of the distribution of 
        pairwise distances between the data-points selected as part of a
        given subsample is stored into a list of medians. 

    Returns
    -------
    preference : float
        The preference parameter for Affinity Propagation clustering is computed
        as the median of the list of median pairwise distances between the data-points
        selected as part of each of 15 rounds of random subsampling.
    """

    N_samples, N_features = data.shape
    
    rng = np.arange(0, N_samples, dtype = int)
    medians = []
    
    for i in range(15):
        selected_samples = np.random.choice(N_samples, size = chunk_size, replace = False)
        samples = data[selected_samples, :]
                
        S = - euclidean_distances(samples, data, squared = True)
                
        n = chunk_size * N_samples - (chunk_size * (chunk_size + 1) / 2)
                
        rows = np.zeros(0, dtype = int)
        for i in range(chunk_size):
            rows = np.append(rows, np.full(N_samples - i, i, dtype = int))
                
        cols = np.zeros(0, dtype = int)
        for i in range(chunk_size):
            cols = np.append(cols, np.delete(rng, selected_samples[:i+1]))
                        
        triu_indices = tuple((rows, cols))
                
        preference = np.median(S, overwrite_input = True)
        medians.append(preference)
                
        del S
                
        if i % 4 == 3:
            gc.collect()       
            
    preference = np.median(medians)

    return preference


def main():

    opts, args = parse_options()
    
    with open(args, 'r') as fh:
        data = np.loadtxt(fh, dtype = float, delimiter = '\t')
        
    N = data.shape[0]
    
    fh = None
    try:
        if opts.hdf5_file is None:
            fh = NamedTemporaryFile('w', delete = True, dir = './',
                                    suffix = '.h5')
            hdf5_file = fh.name
        else:
            hdf5_file = opts.hdf5_file
            fh = open(opts.hdf5_file, 'r+') 
            
        check_HDF5_arrays(hdf5_file, N, opts.convergence_iter)         
        # compute the matrix of pairwise squared Euclidean distances 
        # between all sampled cells and store it in a 
        # HDF5 data structure on disk                           
        if not opts.similarities:
            sim_start = time.time()
            
            compute_similarities(hdf5_file, data, opts.count)
            
            sim_end = time.time()
            
            if opts.verbose:
                delta = sim_end - sim_start
                print(("INFO: concurrent_AP: the computation of the matrix of pairwise "
                      "similarities took {} seconds".format(round(delta, 4))))
        
        if opts.preference is None:
        # The preference is set to the median of all the entries 
        # of the similarities matrix computed at the previous step          
            chunk_size = get_chunk_size(N, 4)
            
            if chunk_size < N:
                preference = set_preference(data, chunk_size)
            else:
                S = - euclidean_distances(data, data, squared = True)
                preference = np.median(S[np.triu_indices(N, k = 1)], 
                                       overwrite_input = True)
                
                del S
                
                gc.collect()
            
        else:
            preference = opts.preference
        
        add_preference(hdf5_file, preference)
        add_fluctuations(hdf5_file, N, opts.count)

        for iteration in range(opts.max_iter):
            iteration_start = time.time()
            
            compute_responsibilities(hdf5_file, N, opts.damping, opts.count)
            rows_sum = compute_rows_sum(hdf5_file, '/aff_prop_group/temporaries',
                                        N, opts.count, method = 'Pool')
            compute_availabilities(hdf5_file, N, opts.damping, 
                                   opts.count, rows_sum)
            # have we reached convergence?
            convergence_flag = check_convergence(hdf5_file, iteration,
                                 opts.convergence_iter, opts.max_iter)
                                 
            iteration_end = time.time()
            
            if opts.verbose:
                delta = iteration_end - iteration_start
                print(("INFO: iteration # {0} took {1} "
                      "seconds".format(iteration + 1, round(delta, 4))))
            
            if convergence_flag:
                break
                
        print('DONE WITH ALL ITERATIONS')
        
        cc_start = time.time()
                
        cluster_centers_indices, labels = get_cluster_labels(hdf5_file,
                                                             opts.count)
        cc_end = time.time()
        
        if opts.verbose:
            delta = cc_end - cc_start
            print(("INFO: the procedure 'get_cluster_labels' "
                  "completed in {} seconds".format(round(delta, 4))))
            
    except EnvironmentError as err:
        print(('ERROR: Affinity_propagation: {0}'.format(err))) 
    finally:
        if fh is not None:
            fh.close()
            
    output_clusters(labels, cluster_centers_indices)


if __name__ == '__main__':

    main()

