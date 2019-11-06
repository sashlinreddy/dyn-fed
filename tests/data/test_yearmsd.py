#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for partitioning algorithm."""

import logging
import unittest

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Need to disable capturing for tensorflow and matplotlib
# before importing from dataset
# pylint: disable=wrong-import-position
from fault_tolerant_ml.data import yearmsd

class TestYearMSD(unittest.TestCase):
    """Tests for Partioning algorithm"""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_yearmsd_load(self):
        """Test yearmsd load
        """
        yearmsd.load_data()
        x = 2

        self.assertEqual(x, 2)
