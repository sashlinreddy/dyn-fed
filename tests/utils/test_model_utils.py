"""Test model related utils
"""
import unittest
import logging
from pathlib import Path

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from dyn_fed.lib.io.file_io import load_yaml
from dyn_fed.utils.model_utils import encode_run_name

logger = logging.getLogger("dfl.utils.tests.utils.test_model_utils")

class TestModelUtils(unittest.TestCase):
    """Test model related utils
    """
    def setUp(self):
        """Set up
        """
        self.config_dir = Path(__file__).resolve().parents[0]

    def tearDown(self):
        """Tear down
        """

    def test_encode_model_name(self):
        """Test encoding of model name
        """
        n_workers = 8
        filename = 'config.yml'
        config_path = self.config_dir/filename
            
        cfg = load_yaml(config_path)

        logger.debug(f"Config={cfg}")

        encoded_run_name = encode_run_name(n_workers, cfg)

        expected_run_name = "8-1-1-0-0-0.0-100-0.0-0-1-0"

        logger.debug(f"Encoded run name={encoded_run_name}, expected={expected_run_name}")

        self.assertEqual(encoded_run_name, expected_run_name)
