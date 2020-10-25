"""Test jinja templating for model config
"""
import unittest
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from dyn_fed.utils.model_utils import render_template

logger = logging.getLogger("dfl.utils.tests.utils.test_model_utils")

class TestConfigTemplating(unittest.TestCase):
    """Test Jinja templating
    """
    def setUp(self):
        """Set up
        """
        self.test_dir = test_dir = Path(__file__).resolve().parents[1]
        self.template_dir = test_dir/"utils/config/template.yml"
        self.config_dir = test_dir/"utils/config/"

    def tearDown(self):
        """Tear down
        """

    def test_jinja_templating(self):
        """Test jinja templating
        """
        output = render_template(
            self.config_dir,
            "template.yml",
            model_version="TF",
            model_type="nn1",
            n_iterations=100,
            check_overfitting=False,
            data_name="mnist",
            noniid=0,
            unbalanced=0,
            learning_rate=0.01,
            optimizer="sgd",
            comm_mode=0,
            interval=1,
            n_workers=8,
            agg_mode=1,
            delta_threshold=0.8,
            data_dir="data/fashion-mnist"
        )

        logger.debug(f"Output={output}")

        with open(self.config_dir/"config.yml", 'r') as f:
            expected_output = f.read()

        logger.debug(f"Expected={expected_output}")

        self.assertEqual(output, expected_output)
