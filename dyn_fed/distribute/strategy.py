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
        self.n_workers = n_workers
        self.config = config
        
        self.tf_dir = config.get('tf_dir')

        self.train_dataset = None
        self.test_dataset = None
        
        self.role = role
        self.identity = config.executor.get('identity')

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

    @property
    def name(self):
        return "master_worker"

    def run(self, *args, **kwargs):
        pass

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
        self.config = config

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