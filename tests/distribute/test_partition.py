#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for partitioning algorithm."""

import logging
import unittest

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Need to disable capturing for tensorflow and matplotlib
# before importing from dataset
# pylint: disable=wrong-import-position
from dyn_fed.data import \
    Dataset
from dyn_fed.data.utils import next_batch, next_batch_unbalanced


class TestPartition(unittest.TestCase):
    """Tests for Partioning algorithm"""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_nooverlap_partition(self):
        """Test partitioning algorithm with no overlap"""
        # Setup data to mirror mnist dataset
        n_samples = 60000
        n_workers = 100
        batch_size = int(np.ceil(n_samples / n_workers))
        overlap = 0.0
        X = np.random.randint(0, n_samples, size=(n_samples, 784))
        y = np.random.randint(0, 2, size=(n_samples, 10))
        generator = next_batch(X, y, batch_size, overlap=overlap)

        for _ in np.arange(n_workers):
            X_batch, y_batch = next(generator)
            # All shapes should be 720 since we have an overlap
            self.assertEqual(X_batch.shape, (600, 784))
            self.assertEqual(y_batch.shape, (600, 10))

    def test_overlap_partition(self):
        """Test partitioning algorithm with overlap"""
        # Setup data to mirror mnist dataset
        n_samples = 60000
        n_workers = 100
        batch_size = int(np.ceil(n_samples / n_workers))
        overlap = 0.2
        X = np.random.randint(0, n_samples, size=(n_samples, 784))
        y = np.random.randint(0, 2, size=(n_samples, 10))
        generator = next_batch(X, y, batch_size, overlap=overlap)

        for _ in np.arange(n_workers):
            X_batch, y_batch = next(generator)
            # All shapes should be 720 since we have an overlap
            self.assertEqual(X_batch.shape, (720, 784))
            self.assertEqual(y_batch.shape, (720, 10))

    def test_unbalanced_dataset(self):
        """Test unbalanced partitioning
        """
        n_samples = 60000
        n_workers = 7
        np.random.seed(2)
        x = np.random.randint(0, n_samples, size=(n_samples, 784))
        y = np.random.randint(0, 2, size=(n_samples, 10))
        batch_gen = next_batch_unbalanced(x, y, n_workers, shuffle=True)

        shapes = []

        for i in np.arange(n_workers):
            x_batch, _ = next(batch_gen)
            shapes.append(x_batch.shape[0])

        assert shapes == [9710, 3350, 14328, 9756, 7752, 1773, 13331]


