"""Logistic regression implementation
"""
import dyn_fed as ft
from dyn_fed.layers import Layer

class LogisticRegressionV2(ft.ModelV2):
    """Logistic regression
    """
    def __init__(
            self,
            optimizer,
            strategy=None,
            batch_size=64,
            max_iter=300,
            shuffle=True,
            verbose=10,
            **kwargs):
        super().__init__(
            optimizer=optimizer,
            strategy=strategy, 
            batch_size=batch_size,
            max_iter=max_iter, 
            shuffle=shuffle, 
            verbose=verbose, 
            **kwargs
        )

        self.add(Layer(784, 10, activation="sigmoid"))
        