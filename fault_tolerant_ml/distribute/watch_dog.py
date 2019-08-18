import logging

class WorkerState(object):
    """Keeps track of workers state for multiple different things.
    
    Has a unique identifier, the state (whether or not the worker is alive), their most representative data points for when they die, the lower and upper bounds for the indices they have, their data indices, the number of samples, and the mapping from the new data point indices to the original data point indices.
    
    Attributes:
        identity (byte string): Unique identifier of the worker
        state (bool): Whether or not the worker is alive
        most_representative (numpy.ndarray): A vector of the indices of the most                representative data points
        lower_bound (int): The lower bound data index the worker contains
        upper_bound (int): The upper bound data index the worker contains
        idxs (numpy.ndarray): The indices the worker contains
        n_samples (int): The number of samples the worker has
        mr_idxs_used (bool): Whether or not a worker's most representative indices have         already been distributed. This applies when to only workers that are dead
        mapping (dict): A mapping from the new data point indices to the original data          indices. This will be the same for workers that haven't died.
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

    def __repr__(self):
        return f"<WorkerState identity={self.identity.decode()}>"

class WorkerStates(object):
    """Wraps dictionary of WorkerState objects to manipulate easily
    """
    def __init__(self):
        self._states = {}

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
        return [s.decode() for s in self._states.keys()]

    def add(self, worker):
        """Add worker state to dictionary
        
        Args:
            worker (byte string): Worker identifier
        """
        self._states[worker] = WorkerState(worker)

    def update_state(self, worker, state):
        self._states[worker] = state

class WatchDog(object):
    """Keeps a watch on all worker related activities
    
    Attributes:
        worker_states (tools.WorkerStates): A dictionary of WorkerState objects
    """
    def __init__(self):
        self._worker_states = WorkerStates()
        self._n_alive = 0

        self.logger = logging.getLogger("ftml")

    @property
    def states(self):
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


    