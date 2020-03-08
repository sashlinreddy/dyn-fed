"""Strategy for distribution
"""
from abc import ABCMeta, abstractmethod
from typing import List
import time
import logging

from dyn_fed.distribute.masterv3 import MasterV3
from dyn_fed.distribute.workerv3 import WorkerV3

class DistributionStrategy(metaclass=ABCMeta):
    """Strategy for distributing within the cluster
    
    Responsible for master-worker strategy. Contains all variables regarding the strategy
    
    Attributes:
        type (int): strategy type
        n_most_rep (int): No. of most representative points
        comm_period (int): Communicate every comm_period iteration(s)
        delta_switch (float): The delta threshold we reach before switching back
        to communication every iteration
    """
    @property
    def name(self):
        """Returns name of strategy
        """
        raise NotImplementedError("Child should override this")

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run strategy
        """
        raise NotImplementedError("Child should override this method")

class LocalStrategy(DistributionStrategy):
    """Local strategy
    """
    def __init__(self, config):
        self.strategy = config.get('strategy')
        self.scenario = config.get('scenario')

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

    @property
    def name(self):
        return "local"

    def run(self, *args, **kwargs):
        pass # TODO: Run local strategy


class MasterWorkerStrategy(DistributionStrategy):
    """Master worker strategy for distributing within the cluster
    
    Attributes:
        n_workers (int): No. of workers being used for session
        remap (int): Remap strategy
        quantize (int): Whether or not to quantize parameters
        comm_period (int): Communicate every comm_period iteration(s)
        aggregate_mode (int): Type of aggregation to be used (weighted average,
        weighted by loss, etc)
        delta_switch (float): The delta threshold we reach before switching back
        to communication every iteration
        worker_timeout (int): Timeout to pickup workers
        send_gradients (int): Whether or not to send gradients or parameters
        shared_folder (str): Path to shared data folder
        role (str): Master or worker role
    """
    def __init__(self, n_workers, config, role='master'):
        if config.get('executor') is not None:
            config = config.get('executor')
        self.strategy = config.strategy
        self.scenario = config.scenario
        self.n_workers = n_workers
        self.comm_period = config.interval
        self.comm_mode = config.mode

        self.remap = config.remap
        self.quantize = config.quantize
        self.overlap = config.overlap
        self.aggregate_mode = config.aggregate_mode
        self.delta_switch = config.delta_switch
        self.worker_timeout = config.timeout
        self.send_gradients = config.send_gradients

        self.shared_folder = config.shared_folder
        self.config_folder = config.config_folder
        self.tf_dir = config.get('tf_dir')
        self.unbalanced = config.unbalanced
        self.norm_epsilon = config.norm_epsilon

        self.train_dataset = None
        self.test_dataset = None
        
        self.role = role
        # if self.role == 'worker':
        self.identity = config.get('identity')

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

    @property
    def name(self):
        return "master_worker"

    def run(self, *args, **kwargs):
        pass
class objectify(dict):
    """Class to wrap dictionaries and make them more accessible
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, objectify):
            value = objectify(value)
        super(objectify, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, objectify.MARKER)
        if found is objectify.MARKER:
            found = objectify()
            super(objectify, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__

class MasterWorkerStrategyV2(DistributionStrategy):
    """Master worker strategy for distributing within the cluster
    
    Attributes:
        n_workers (int): No. of workers being used for session
        remap (int): Remap strategy
        quantize (int): Whether or not to quantize parameters
        comm_period (int): Communicate every comm_period iteration(s)
        aggregate_mode (int): Type of aggregation to be used (weighted average,
        weighted by loss, etc)
        delta_switch (float): The delta threshold we reach before switching back
        to communication every iteration
        worker_timeout (int): Timeout to pickup workers
        send_gradients (int): Whether or not to send gradients or parameters
        shared_folder (str): Path to shared data folder
        role (str): Master or worker role
    """
    def __init__(self, n_workers, config, role='master'):
        self.n_workers = n_workers
        self.config = objectify(config)

        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.optimizer = None
        self.train_step = None
        self.test_step = None
        self.role = role

        self.identity = self.config.executor.identity

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

    @property
    def name(self):
        return "master_worker"

    def distribute_dataset(self, dataset):
        """Distribute dataset
        """
        if self.identity is not None:
            dataset = dataset.shard(
                dataset,
                self.identity
            )
        else:
            return dataset

    def run(self, *args, **kwargs):
        model = args[0]
        optimizer = args[1]
        train_dataset = args[2]
        test_dataset = kwargs.get('test_dataset')
        if self.role == "master":
            # Setup master
            self._logger.debug("Setting up master")
            self._master = MasterV3(
                model=model,
                optimizer=optimizer,
                strategy=self
            )
            self._master.setup(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
            self._master.start()
        else:

            time.sleep(3)

            self._worker = WorkerV3(
                model=model,
                optimizer=optimizer,
                strategy=self
            )

            self._worker.setup(
                train_dataset=None
            )

            self._worker.start()

            self._logger.info("Connecting worker sockets")