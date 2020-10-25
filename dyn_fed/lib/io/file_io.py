"""Functions to accomplish basic File IO operations.
"""
from __future__ import absolute_import, division, print_function

import datetime
import logging
import os
import time
from pathlib import Path

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from dyn_fed.utils.general_utils import objectify

logger = logging.getLogger("dfl.lib.io.file_io")

def get_project_dir():
    p = Path(__file__).resolve().parents[3]
    return p

def flush_dir(dir, ignore_dir=[], mins=1, hours=0):
    """Flushes directory for files that were modified hours, mins ago

    Args:
        ignore_dir (list): List of directories to exclude from being flushed
        mins (int): No. of minutes ago that serves as a cutoff for which files we exclude       from being flushed
        hours (int): No. of hours ago that serves as a cutoff for which files we exclude        from being flushed
    """
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            curpath = os.path.join(dirpath, file)

            # Get time of last time the file was modified
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(curpath))
            
            # Ignore files that were created less than hours, mins ago
            if datetime.datetime.now() - file_modified > datetime.timedelta(minutes=mins, hours=hours):
                # Ignore files in a certain directory
                ignore_files_length = len([dir for dir in ignore_dir if dir in dirpath])
                if (ignore_files_length < 1) or len(ignore_dir) > 0:
                    os.remove(curpath)

def load_yaml(path, to_obj=True):
    """Load model config given path

    Args:
        path (str): Path to model config file

    Returns:
        cfg (dict): Config dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if to_obj:
        return objectify(cfg)
    else:
        return cfg

def export_yaml(data, path):
    """Export model config given path

    Args:
        path (str, pathlib.Path): Path to model config file
    """
    path = Path(path)
    if not path.is_dir():
        path.parents[0].mkdir(parents=True, exist_ok=True)
    elif path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        
    with open(path, 'w') as ymlfile:
        yaml.dump(data, ymlfile, default_flow_style=False)

class Handler(FileSystemEventHandler):
    """Watchdog handler
    """
    def __init__(self, watcher):
        self.watcher = watcher
        self.observer = self.watcher.observer
        self.filename = self.watcher.filename

    def on_any_event(self, event):
        """Overridden on any event method
        """
        if event.is_directory:
            return None

        if event.event_type == 'created':
            # Take any action here when a file is first created.
            logger.debug("Received created event - %s.", event.src_path)
            if event.src_path == self.filename:
                self.watcher.file_found = True
                self.observer.stop()

        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            logger.debug("Received modified event - %s.", event.src_path)

class FileWatcher:
    """File watching class
    """
    def __init__(self, dir_to_watch, filename):
        self.observer = Observer()
        self.file_found = False
        self.filename = filename
        self.dir_to_watch = dir_to_watch

    def run(self, timeout=20, recursive=False):
        """Run file watcher
        """
        start = time.time()
        running_time = 0
        event_handler = Handler(self)
        logger.debug(f"Watching {self.dir_to_watch}")
        self.observer.schedule(event_handler, self.dir_to_watch, recursive=recursive)
        self.observer.start()
        try:
            while not self.file_found:
                running_time = time.time() - start
                if running_time > timeout:
                    self.observer.stop()
                    logger.debug("Timed out, no file found.")
                    break
            logger.debug(f"File {self.filename} found!")
        except KeyboardInterrupt:
            self.observer.stop()
            logger.warning("Ctrl-c pressed. Exiting")

        self.observer.join()

        return self.file_found
