#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fault_tolerant_ml` package."""


import unittest
from click.testing import CliRunner

from fault_tolerant_ml import cli


class TestFault_tolerant_ml(unittest.TestCase):
    """Tests for `fault_tolerant_ml` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_pwd_dotenv(self):
        """Test pwd in dotenv file"""
        from dotenv import load_dotenv, find_dotenv
        from pathlib import Path
        import os

        load_dotenv(find_dotenv())
        project_dir = Path(__file__).resolve().parents[1]
        logdir = os.path.join(project_dir, os.environ["LOGDIR"])
        print(f"project_dir={project_dir}, logdir={logdir}")

        assert logdir == os.path.join(project_dir, "logs/")
