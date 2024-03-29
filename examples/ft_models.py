"""Example models that can be implemented using dfl
"""
import dyn_fed as ft
from dyn_fed.layers import Layer


class MLP(ft.Model):
    """Multilayer perceptron
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

        self.add(Layer(784, 128, activation="sigmoid"))
        self.add(Layer(128, 10, activation="sigmoid"))

class LogisticRegression(ft.Model):
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

class SimpleNN(ft.ModelV2):
    """Multilayer perceptron
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

        self.add(Layer(784, 128, activation="sigmoid"))
        self.add(Layer(128, 10, activation="sigmoid"))

class NN2(ft.ModelV2):
    """Multilayer perceptron
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

        self.add(Layer(784, 256, activation="sigmoid"))
        self.add(Layer(256, 128, activation="sigmoid"))
        self.add(Layer(128, 10, activation="sigmoid"))

class LinearRegression(ft.Model):
    """Linear regression
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

        self.add(Layer(784, 10, activation="linear"))

class LinearRegressionV2(ft.ModelV2):
    """Linear regression
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

        self.add(Layer(784, 10, activation="linear"))
