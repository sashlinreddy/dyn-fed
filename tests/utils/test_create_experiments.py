"""Test jinja templating for model config
"""
import unittest
import logging
from pathlib import Path

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from dyn_fed.utils.model_utils import gen_exp_perms
from dyn_fed.lib.io import file_io

logger = logging.getLogger("dfl.utils.tests.utils.test_model_utils")

class TestConfigTemplating(unittest.TestCase):
    """Test Jinja templating
    """
    def setUp(self):
        """Set up
        """
        self.test_dir = test_dir = Path(__file__).resolve().parents[1]
        self.project_dir = test_dir/"utils/"
        self.experiments_path = self.project_dir/'config/experiments.yml'

    def tearDown(self):
        """Tear down
        """

    def test_create_experiments(self):
        """Test jinja templating
        """
        experiments_cfg = file_io.load_yaml(self.experiments_path)
        print(experiments_cfg)
        experiments = gen_exp_perms(experiments_cfg)
        self.assertEqual(len(experiments), 1760)
