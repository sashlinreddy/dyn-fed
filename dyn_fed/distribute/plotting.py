"""Plot metrics related to distributed environment
"""
import logging
import os

from dyn_fed.viz.target import ClassBalance

logger = logging.getLogger("dfl.distribute.plotting")

def plot_distributed_class_balance(x, y, class_names, filepath, **kwargs):
    """Plot distributed computing related metrics

    Args:
        x (list): List of worker ids
        y (list): Count of labels per each worker
        class_names (list): List of class names
        filepath (str): File path to save image
        kwargs (dict):
            tf_logger (tf.Logger): Logger for tensorboard
            iteration (int): Iteration for tensorboard
    """
    # if "FIGDIR" in os.environ:
    tf_logger = kwargs.get("tf_logger")
    iteration = kwargs.get("iteration", 0)

    # figdir = os.path.join(os.environ["FIGDIR"], encode_name)
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    logger.debug("Saving class balances distribution plot...")
    # worker_ids = [s.identity.decode() for s in self.watch_dog.states if s.state]
    # class_bal = [
    #     v[1] for (k, v) in self.coordinator.labels_per_worker.items()
    #     if k.identity.decode() in worker_ids
    # ]
    fname = os.path.join(filepath, f"mnist-class-balance.png")

    class_balance = ClassBalance(
        labels=x,
        legend=class_names,
        fname=fname,
        stacked=True,
        percentage=True
    )
    class_balance.fit(y=y)
    class_balance.poof()

    fig = class_balance.fig

    if tf_logger is not None:
        tf_logger.images("class-bal-server", [fig], iteration)
