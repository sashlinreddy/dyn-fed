"""Watch dog for clients joining and leaving
"""
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

class WorkerState():
    """Keeps track of clients state for multiple different things.
    
    Has a unique identifier, the state (whether or not the client is alive),
    their most representative data points for when they die, the lower and upper
    bounds for the indices they have, their data indices, the number of samples,
    and the mapping from the new data point indices to the original data point
    indices.
    
    Attributes:
        identity (byte string): Unique identifier of the client
        state (bool): Whether or not the client is alive
        most_representative (numpy.ndarray): A vector of the indices of the
        most representative data points
        lower_bound (int): The lower bound data index the client contains
        upper_bound (int): The upper bound data index the client contains
        idxs (numpy.ndarray): The indices the client contains
        n_samples (int): The number of samples the client has
        mr_idxs_used (bool): Whether or not a client's most representative
        indices have already been distributed. This applies when to only clients
        that are dead
        mapping (dict): A mapping from the new data point indices to the original
        data indices. This will be the same for clients that haven't died.
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
        """Updates client comm period
        """
        self._comm_iterations = comm_iterations

    @property
    def comm_interval(self):
        """Returns comm period
        """
        return self._comm_interval

    @comm_interval.setter
    def comm_interval(self, comm_interval):
        """Updates client comm intervals
        """
        self._comm_interval = comm_interval
    
    @property
    def comm_every_iter(self):
        """Returns comm every iter
        """
        return self._comm_every_iter

    @comm_every_iter.setter
    def comm_every_iter(self, comm_every_iter):
        """Updates client comm every iter
        """
        self._comm_every_iter = comm_every_iter

    @property
    def comm_rounds(self):
        """Returns comm every iter
        """
        return self._comm_rounds

    @comm_rounds.setter
    def comm_rounds(self, comm_rounds):
        """Updates client comm every iter
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
        """Returns client state keys
        """
        return [s.decode() for s in self._states]

    def add(self, client):
        """Add client state to dictionary
        
        Args:
            client (byte string): Worker identifier
        """
        self._states[client] = WorkerState(client)

    def pop(self, client):
        """Remove client from dictionary
        
        Args:
            client (byte string): Worker identifier
        """
        self._states.pop(client, None)

    def update_state(self, client, state):
        """Updates client state
        """
        self._states[client] = state

class WatchDog(object):
    """Keeps a watch on all client related activities
    
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
        """Returns client states
        """
        return self._worker_states

    @property
    def n_alive(self):
        """Returns number of clients that are alive
        
        Returns:
            n_alive (int): No. of alive clients
        """
        self._n_alive = sum([s.state for s in self._worker_states().values()])
        return self._n_alive

    @property
    def active_workers(self):
        """Returns activate clients
        """
        return [w.identity for w in self.states if w.state]

    def add_worker(self, client):
        """Adds new client to dictonary of clients
        
        Args:
            client (byte str): Unique identifier for client
        """
        if client not in self.states():
            self.logger.info(f"Worker Registered: {client}")
            self.states.add(client)
        elif not self.states()[client].state:
            self.logger.info(f"Worker {client} alive again")
            self.states[client].state = True
        else:
            self.logger.debug("Worker asking for work again?")

    def pop(self, client):
        """Remove client
        """
        self.logger.info(f"Removing client {client} due to heart failure :(")
        self._worker_states.pop(client)

class ModelWatchDog():
    """Watchdog for model related things
    """
    def __init__(self,
                 model: tf.keras.Sequential,
                 n_samples: int,
                 n_classes: int,
                 delta_threshold=0.8):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.delta_threshold = delta_threshold
        self.ref_model = [p.numpy() for p in model.trainable_weights]
        self.prev_model = None
        self._divergence = np.inf

    def update_ref_model(self, model):
        """Update reference model
        """
        self.prev_model = self.ref_model
        self.ref_model = [p.numpy() for p in model.trainable_weights]
        self.calculate_divergence()

    def calculate_divergence(self):
        """Calculate divergence
        """
        self._divergence = np.max([
            np.linalg.norm(o - n)**2
            for o, n in zip(self.prev_model, self.ref_model)
        ])

    def model_condition(self):
        """Check global condition.

        If divergence < delta_threshold then we have have not violated the condition
        and no communication is needed
        """
        return self.divergence < self.delta_threshold

    @property
    def divergence(self):
        """Property getter for divergence
        """
        return self._divergence
