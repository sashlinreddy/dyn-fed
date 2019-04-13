import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ClassBalance(object):
    """Plot class balance given the class labels
    
    The ClassBalance visualizer can be displayed in two modes:

    1. Balance mode: show the frequency of each class in the dataset.
    2. Compare mode: show the relationship of support across different data partitions.

    These modes are determined by what is passed to the `fit()` method.
    
    Attributes:
        ax (matplotlib Axes): The axis to plot the figure on. If None is passed in the current axes will be used (or            generated if required) (default: None)
        labels (list): List of class names for the x-axis if the target is already encoded. In compare mode, this will be a     list of different data partitions (optional)
        yticks (np.ndarray): yticks for matplotlib plot
        fname (str): Filename to save plot
        dpi (int): Dots per inch. (default: 500)
        title (str): Title for plot
        xlabel (str): X-axis label for plot
        ylabel (str): Y-axis label for plot
        figsize (tuple): Size of figure as a tuple (x, y)
        stacked (bool): Whether or not to stack bars. Used in compare mode
        percentage (bool): Whether or not to have a 100% stacked bar chart
    """
    def __init__(self, ax=None, labels=None, **kwargs):
        
        self.ax = ax
        self.labels = labels
        self.yticks = None
        self.fname = None
        self.dpi = 500
        self.title = "Class balance distribution"
        self.xlabel = "Class"
        self.ylabel = "Count"
        self.figsize = (16, 9)
        self.stacked = False
        self.percentage = False

        if "yticks" in kwargs:
            self.yticks = kwargs["yticks"]

        if "fname" in kwargs:
            self.fname = kwargs["fname"]

        if "dpi" in kwargs:
            self.dpi = kwargs["dpi"]

        if "title" in kwargs:
            self.title = kwargs["title"]

        if "xlabel" in kwargs:
            self.xlabel = kwargs["xlabel"]

        if "ylabel" in kwargs:
            self.ylabel = kwargs["ylabel"]

        if "figsize" in kwargs:
            self.figsize = kwargs["figsize"]

        if "stacked" in kwargs:
            self.stacked = kwargs["stacked"]

        if "percentage" in kwargs:
            self.percentage = kwargs["percentage"]

        if "legend" in kwargs:
            self.legend = kwargs["legend"]

    def fit(self, y, **kwargs):
        """Fit the visualizer to the the target variables

        1. Balance mode: show the frequency of each class in the dataset.
        2. Compare mode: show the relationship of support across different data partitions.

        In balance mode, the bar chart is displayed with each class as its own color. In compare mode, a stacked bar chart is displayed colored by the different data partitions.

        Args:
            y (list): List of class balances. If len(y) > 1, then we are in compare mode
            kwargs: Any other keyword arguments
        """
        if not isinstance(y, list):
            y = [y]

        # Compare mode
        if len(y) > 1:
            self.labels_distribution = pd.DataFrame(y, index=self.labels, columns=self.legend)
            if self.percentage:
                totals = self.labels_distribution.sum(axis=1).values
                self.labels_distribution = self.labels_distribution.div(totals, 0) * 100
                self.ylabel = "Percentage %"
        # Balance mode
        # TODO: Test balance mode
        else:
            self.labels_distribution = pd.Series(y, index=self.labels)


    def poof(self, **kwargs):
        """Displays class imbalances or returns axes
        """
        # Create subplot
        if self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
    
        # Use pandas dataframe built in plot to easily show stacked bars
        self.labels_distribution.plot(ax=self.ax, kind="bar", stacked=self.stacked)

        # Set figure title, axis labels, etc
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.yticks is not None:
            self.ax.set_yticks(self.yticks)
        
        # Save plot
        if self.fname is not None:
            plt.savefig(fname=self.fname, dpi=self.dpi)

        return self.ax