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
        logdir = os.environ["LOGDIR"]
        print(f"project_dir={project_dir}, logdir={logdir}")

        assert logdir == os.path.join(project_dir, "logs")

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'fault_tolerant_ml.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
