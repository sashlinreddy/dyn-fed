#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for shapley contribution algorithm."""

import logging
import unittest

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Need to disable capturing for tensorflow and matplotlib
# before importing from dataset
# pylint: disable=wrong-import-position
from dyn_fed.distribute.watch_dog import SHAPWatchDog


class TestPartition(unittest.TestCase):
    """Tests for Partioning algorithm"""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_shapley_values(self):
        """Test shapley contributions"""
        clients = [
            'client-1',
            'client-2',
            'client-3',
            'client-4'
        ]
        shap_watchdog = SHAPWatchDog(clients)

        losses = [8, 8, 8, 1, 3, 4, 6, 5, 5, 1, 8, 9, 2, 0, 4]

        self.assertEqual(shap_watchdog.client_int.tolist(), [0, 1, 2, 3])

        for i in range(len(shap_watchdog.pset)):
            clients = shap_watchdog.pset[i]
            print(shap_watchdog.clients[clients])
            # Average parameters amongst these clients
            # Get loss from model
            # Store as characteristic function for this combination
            shap_watchdog.set_charact_fn(i, losses[i])

        shapley_values = shap_watchdog.shapley_values()
        print(shapley_values)

        print(np.exp(shapley_values) / np.sum(np.exp(shapley_values)))

        assert False
        