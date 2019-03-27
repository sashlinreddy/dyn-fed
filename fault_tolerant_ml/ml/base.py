from .metrics_temp import accuracy_score

class BaseEstimator(object):

    def __init__(self):
        pass

    def funcname(self, parameterlist):
        pass

class ClassifierMixin(object):

    def __init__(self):
        pass

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class RegressorMixin(object):

    def __init__(self):
        pass

    def score(self, X, y):
        pass