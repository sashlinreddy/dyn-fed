"""Test jinja templating for model config
"""
import unittest
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from fault_tolerant_ml.utils.model_utils import render_template

logger = logging.getLogger("ftml.utils.tests.utils.test_model_utils")

class TestConfigTemplating(unittest.TestCase):
    """Test Jinja templating
    """
    def setUp(self):
        """Set up
        """
        self.test_dir = test_dir = Path(__file__).resolve().parents[1]
        self.template_dir = test_dir/"utils/template.yml"
        self.config_dir = test_dir/"utils/"

    def tearDown(self):
        """Tear down
        """

    def test_jinja_templating(self):
        """Test jinja templating
        """
        output = render_template(
            self.config_dir,
            "template.yml",
            n_iterations=100,
            comm_mode=0,
            interval=1,
            agg_mode=0,
            data_dir="\"data/fashion-mnist/\"",
            noniid=0,
            unbalanced=0
        )

        logger.debug(f"Output={output}")

        with open(self.config_dir/"config.yml", 'r') as f:
            expected_output = f.read()

        logger.debug(f"Expected={expected_output}")

        self.assertEqual(output, expected_output)
