import fault_tolerant_ml as ft
from fault_tolerant_ml.layers import Layer

class MLP(ft.Model):

    def __init__(self, strategy=None, **kwargs):
        super().__init__(strategy, **kwargs)

        self.add(Layer(784, 32, activation="sigmoid"))
        self.add(Layer(32, 10, activation="sigmoid"))

class LogisticRegression(ft.Model):

    def __init__(self, strategy=None, **kwargs):
        super().__init__(strategy, **kwargs)

        self.add(Layer(784, 10, activation="sigmoid"))