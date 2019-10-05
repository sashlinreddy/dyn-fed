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
from fault_tolerant_ml.data import \
    Dataset
from fault_tolerant_ml.data.utils import next_batch


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
