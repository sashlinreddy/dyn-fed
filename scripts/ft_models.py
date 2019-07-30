import fault_tolerant_ml as ft
from fault_tolerant_ml.layers import Layer

class MLP(ft.Model):

    def __init__(self, optimizer, strategy=None, max_iter=300, shuffle=True, verbose=10, **kwargs):
        super().__init__(optimizer, strategy, max_iter, shuffle, verbose, **kwargs)

        self.add(Layer(784, 32, activation="sigmoid"))
        self.add(Layer(32, 10, activation="sigmoid"))

class LogisticRegression(ft.Model):

    def __init__(self, optimizer, strategy=None, max_iter=300, shuffle=True, verbose=10, **kwargs):
        super().__init__(optimizer, strategy, max_iter, shuffle, verbose, **kwargs)

        self.add(Layer(784, 10, activation="sigmoid"))