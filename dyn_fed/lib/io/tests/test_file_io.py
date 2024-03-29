#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os

from dyn_fed.lib.io.file_io import flush_dir

class TestFileIO(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_flush_dir(self):
        """Test flushing a dir"""
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)
        files = os.listdir(os.environ["LOGDIR"])
        files = [f for f in files if f not in ["tf"]]
        print(f"Files={files}")
        assert len(files) < 1

    def test_flush_dir_no_ignore(self):
        """Test flushing a dir without ignoring any directories"""
        ignore_dir = []
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)
        files = os.listdir(os.environ["LOGDIR"])
        print(f"Files={files}")
        assert len(files) < 1
