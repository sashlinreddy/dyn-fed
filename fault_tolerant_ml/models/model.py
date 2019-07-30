"""This file contains all logic for the base class of models. For implementation of logistic regression or neural networks,
you should inherit from this class.
"""

import logging
import time

from fault_tolerant_ml.distribute import Master, Worker
from fault_tolerant_ml.operators import Tensor
from fault_tolerant_ml.layers import Layer
from fault_tolerant_ml.distribute.strategy import LocalStrategy

def _check_layers(layers):
    """Check if all objects in list are of Layer type

    Args:
        layers (list): List of fault_tolerant_ml.layers.Layer objects

    Returns:
        valid (bool): Whether or not all layers are valid
    """
    valid = True
    type_found = None
    for layer in layers:
        if not isinstance(layer, Layer):
            valid = False
            type_found = type(layer)
            break

    return valid, type_found

# strategy=None, **kwargs):
        
#         self.encode_name = None
#         if "encode_name" in kwargs:
#             self.encode_name = kwargs["encode_name"]
#         self.optimizer = optimizer

#         if strategy is None:
#             self.strategy = LocalStrategy(config={})
#         else:
#             self.strategy = strategy

#         self._logger = logging.getLogger(f"ftml.models.{self.__class__.__name__}")

#     def encode(self):
#         return f"{self.strategy.n_workers}-{self.strategy.scenario}-{self.strategy.remap}-{self.strategy.quantize}-{self.optimizer.n_most_rep}-{self.strategy.comm_period}-{self.optimizer.mu_g}-{self.strategy.send_gradients}"

class Model():
    """
    Base class for tensor models.

    To create your own model, use as follows:

    Example:
        class LogisticRegression(fault_tolerant.Model):

            def __init__(self, strategy=None, **kwargs):
                super().__init__(strategy, **kwargs)

                self.add(Layer(784, 10), activation_fn='sigmoid')

        lr = LogisticRegression()

        y_pred = lr.forward(X_train)

        loss = loss(y, y_pred)

    Attributes:
        layers (list): List of fault_tolerant_ml.layers.Layer objects
    """
    def __init__(self, optimizer, strategy=None, max_iter=300, shuffle=True, verbose=10, **kwargs):
        self.layers = []
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.iter = 0
        self.verbose = verbose
        self.optimizer = optimizer

        self.encode_name = None
        if "encode_name" in kwargs:
            self.encode_name = kwargs["encode_name"]

        if strategy is None:
            self.strategy = LocalStrategy(config={})
        else:
            self.strategy = strategy

        self._logger = logging.getLogger(f"ftml.models.{self.__class__.__name__}")

        # Setup distribution strategy
        self._setup()
        
    def __repr__(self):
        return f"Model([{self.layers}])"

    def _setup(self):
        """Setup distribution strategy
        """
        if self.strategy.name == "local":
            pass # TODO: Local strategy
        elif self.strategy.name == "master_worker":
            if self.strategy.role == "master":
                # Setup master
                self._master = Master(
                    model=self,
                    verbose=self.verbose,
                )

            else:

                time.sleep(2)

                self._worker = Worker(
                    model=self,
                    verbose=self.verbose,
                    id=self.strategy.identity
                )

    def _fit_mw(self, X=None, y=None):
        """Training logistic regression using the master worker strategy

        Args:
            X (fault_tolerant.operators.Tensor): Feature dataset
            y (fault_tolerant.operators.Tensor): Labels
        """
        if self.strategy.role == "master":
            # Master training
            self._logger.info("Connecting master sockets")
            self._master.connect()
                # setattr(master, "train_iter", train_iter)
                # time.sleep(1)
            self._master.train(X)
        else:
            self._logger.info("Connecting worker sockets")
            self._worker.connect()
            # Worker training
            self._worker.train()
        
    def add(self, layers):
        """Add new layer(s) to model
        """

        if not isinstance(layers, list):
            layers = [layers]
        valid, type_found = _check_layers(layers)
        assert valid, f"All layers should be of type Layer, found {type_found} instead"
        self.layers += layers
            
    def zero_grad(self):
        """Zeros out the gradients
        """
        for layer in self.layers:
            for _, v in layer.__dict__.items():
                if isinstance(v, Tensor):
                    if v.is_param:
                        v.zero_grad()

    def compile(self, optimizer):
        """Compile the model with the corresponding optimizer
        """
        self.optimizer = optimizer

    def fit(self, X=None, y=None):
        """Training for estimating parameters
        """
        if self.strategy.name == 'local':
            pass # TODO: Add local strategy
        elif self.strategy.name == "master_worker":
            self._fit_mw(X, y)
            
    def forward(self, x):
        """Feedforward through network given input x

        Args:
            x (fault_tolerant_ml.operators.Tensor): Input tensor which is the feature dataset

        Returns:
            y_pred (fault_tolerant_ml.operators.Tensor): Output tensor which is the prediction for each class
        """
        input_layer = x
        for layer in self.layers:
            y_pred = layer(input_layer)
            input_layer = layer.y
            
        return y_pred