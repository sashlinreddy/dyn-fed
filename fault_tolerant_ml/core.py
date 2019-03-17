class WorkerState(object):
    """Short description
    
    Long description
    
    Attributes:
        attrib1 (type): Short description of attribute
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
    """Short description
    
    Long description
    
    Attributes:
        attrib1 (type): Short description of attribute
    """
    def __init__(self):
        self._states = {}

    def __getitem__(self, key):
        return self._states[key]

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        for key, value in self._states.items():
            yield value

    def __contains__(self, key):
        return key in self._states

    def __repr__(self):
        rep = ", ".join([s.identity.decode() for s in self._states.values()])
        rep = f"[{rep}]"
        return rep

    def add_worker(self, worker):
        """Adds new worker to dictonary of workers
        
        Long description
        
        Args:
            worker (byte str): Unique identifier for worker
        """
        if worker not in self._states:
            self._states[worker] = WorkerState(worker)

    def n_alive(self):
        """Returns number of workers that are alive
        
        Returns:
            n_alive (int): No. of alive workers
        """
        return sum([s.state for s in self._states.values()])