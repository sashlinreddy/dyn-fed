"""Log scalars, histograms and images to tensorboard without tensor ops.

From https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""
from __future__ import absolute_import, division, print_function

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class TFLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            buf = BytesIO()
            # plt.imsave(s, img, format='png')
            plt.savefig(buf, format="png")

            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(img)
            buf.seek(0)

            size = img.get_size_inches()*img.dpi

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue(),
                                       height=int(size[0]),
                                       width=int(size[1]))

            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        

    def histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # pylint: disable=line-too-long
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge) # pylint: disable=no-member
        for c in counts:
            hist.bucket.append(c) # pylint: disable=no-member

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
