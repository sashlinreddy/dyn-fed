"""Test model related utils
"""
import unittest
import logging
from pathlib import Path

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from dyn_fed.lib.io.file_io import get_project_dir

logger = logging.getLogger("dfl.utils.tests.utils.test_fileio")

class TestFileIO(unittest.TestCase):
    """Test model related utils
    """
    def setUp(self):
        """Set up
        """

    def tearDown(self):
        """Tear down
        """

    def test_project_dir(self):
        """Test valid project dir
        """
        project_dir = get_project_dir()
        expected_project_dir = '/Users/sashlinreddy/repos/dyn-fed'

        self.assertEqual(str(project_dir), expected_project_dir)
        
