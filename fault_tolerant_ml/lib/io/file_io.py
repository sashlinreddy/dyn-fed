"""Functions to accomplish basic File IO operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

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
                if not len([dir for dir in ignore_dir if dir in dirpath]):
                    os.remove(curpath)