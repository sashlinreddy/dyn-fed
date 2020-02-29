"""Watch dog for workers joining and leaving
"""
from __future__ import print_function

import logging

class WorkerState():
    """Keeps track of workers state for multiple different things.
    
    Has a unique identifier, the state (whether or not the worker is alive),
    their most representative data points for when they die, the lower and upper
    bounds for the indices they have, their data indices, the number of samples,
    and the mapping from the new data point indices to the original data point
    indices.
    
    Attributes:
        identity (byte string): Unique identifier of the worker
        state (bool): Whether or not the worker is alive
        most_representative (numpy.ndarray): A vector of the indices of the
        most representative data points
        lower_bound (int): The lower bound data index the worker contains
        upper_bound (int): The upper bound data index the worker contains
        idxs (numpy.ndarray): The indices the worker contains
        n_samples (int): The number of samples the worker has
        mr_idxs_used (bool): Whether or not a worker's most representative
        indices have already been distributed. This applies when to only workers
        that are dead
        mapping (dict): A mapping from the new data point indices to the original
        data indices. This will be the same for workers that haven't died.
    """
    def __init__(self, identity):
        self.identity = identity
        self.state = True
        self.most_representative = None
        self.lower_bound = None
        self.upper_bound = None
        self.idxs = None
        self.n_samples = None
        self.mr_idxs_used = False
        self.mapping = {}
        self.svd_idx = 0
        self._prev_loss = 1.0
        self._current_loss = 1.0
        self._comm_iterations = 1
        self._comm_interval = 1
        self._comm_every_iter = 1
        self._comm_rounds = 0

    def __repr__(self):
        return f"<WorkerState identity={self.identity.decode()}>"

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.identity == other.identity

    @property
    def comm_iterations(self):
        """Returns comm period
        """
        return self._comm_iterations

    @comm_iterations.setter
    def comm_iterations(self, comm_iterations):
        """Updates worker comm period
        """
        self._comm_iterations = comm_iterations

    @property
    def comm_interval(self):
        """Returns comm period
        """
        return self._comm_interval

    @comm_interval.setter
    def comm_interval(self, comm_interval):
        """Updates worker comm intervals
        """
        self._comm_interval = comm_interval
    
    @property
    def comm_every_iter(self):
        """Returns comm every iter
        """
        return self._comm_every_iter

    @comm_every_iter.setter
    def comm_every_iter(self, comm_every_iter):
        """Updates worker comm every iter
        """
        self._comm_every_iter = comm_every_iter

    @property
    def comm_rounds(self):
        """Returns comm every iter
        """
        return self._comm_rounds

    @comm_rounds.setter
    def comm_rounds(self, comm_rounds):
        """Updates worker comm every iter
        """
        self._comm_rounds = comm_rounds

    @property
    def prev_loss(self):
        """Returns previous loss
        """
        return self._prev_loss

    @prev_loss.setter
    def prev_loss(self, prev_loss):
        """Updates previous loss
        """
        self._prev_loss = prev_loss

    @property
    def current_loss(self):
        """Returns current loss
        """
        return self._current_loss

    @current_loss.setter
    def current_loss(self, current_loss):
        """Updates current loss
        """
        self._current_loss = current_loss

class WorkerStates(object):
    """Wraps dictionary of WorkerState objects to manipulate easily
    """
    def __init__(self):
        self._states = {}

        self.logger = logging.getLogger(
            f"dfl.distribute.{self.__class__.__name__}"
        )

    def __call__(self):
        return self._states

    def __getitem__(self, key):
        return self._states[key]

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        for _, value in self._states.items():
            yield value

    def __contains__(self, key):
        return key in self._states

    def __repr__(self):
        rep = ", ".join([s.identity.decode() for s in self._states.values()])
        rep = f"[{rep}]"
        return rep

    def keys(self):
        """Returns worker state keys
        """
        return [s.decode() for s in self._states]

    def add(self, worker):
        """Add worker state to dictionary
        
        Args:
            worker (byte string): Worker identifier
        """
        self._states[worker] = WorkerState(worker)

    def pop(self, worker):
        """Remove worker from dictionary
        
        Args:
            worker (byte string): Worker identifier
        """
        self._states.pop(worker, None)

    def update_state(self, worker, state):
        """Updates worker state
        """
        self._states[worker] = state

class WatchDog(object):
    """Keeps a watch on all worker related activities
    
    Attributes:
        worker_states (tools.WorkerStates): A dictionary of WorkerState objects
    """
    def __init__(self):
        self._worker_states = WorkerStates()
        self._n_alive = 0

        self.logger = logging.getLogger(
            f"dfl.distribute.{self.__class__.__name__}"
        )

    @property
    def states(self):
        """Returns worker states
        """
        return self._worker_states

    @property
    def n_alive(self):
        """Returns number of workers that are alive
        
        Returns:
            n_alive (int): No. of alive workers
        """
        self._n_alive = sum([s.state for s in self._worker_states().values()])
        return self._n_alive

    @property
    def active_workers(self):
        """Returns activate workers
        """
        return [w.identity for w in self.states if w.state]

    def add_worker(self, worker):
        """Adds new worker to dictonary of workers
        
        Args:
            worker (byte str): Unique identifier for worker
        """
        if worker not in self.states():
            self.logger.info(f"Worker Registered: {worker}")
            self.states.add(worker)
        elif not self.states()[worker].state:
            self.logger.info(f"Worker {worker} alive again")
            self.states[worker].state = True
        else:
            self.logger.debug("Worker asking for work again?")

    def pop(self, worker):
        """Remove worker
        """
        self.logger.info(f"Removing worker {worker} due to heart failure :(")
        self._worker_states.pop(worker)
