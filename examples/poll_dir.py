from dyn_fed.lib.io.file_io import FileWatcher
import logging
import sys

if __name__ == '__main__':
    logger = logging.getLogger('dfl')
    formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s() - %(levelname)s - %(message)s',
                                  "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    w = FileWatcher(dir_to_watch="data/", filename="data/hello.txt")
    file_found = w.run(timeout=20)
    if file_found:
        print("File found")
    else:
        print("File not found")