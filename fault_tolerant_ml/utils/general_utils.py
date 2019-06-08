import sys
import logging
import pprint
import os
import shutil as sh

pp = pprint.PrettyPrinter()

def log(msg):
    """
    This is mainly for MPI printing. Stdout needs to be flushed
    """
    print(msg)
    sys.stdout.flush()

def log_begin_iteration(iteration):
    """
    Logs beginning of iteration

    Args:
        iteration: int
            Iteration which we print the beginning of
    """
    log("-" * 40)
    log(f"START OF ITERATION {iteration}")
    log("-" * 40)

def log_end_iteration(iteration):
    """
    Logs end of iteration

    Args:
        iteration: int
            Iteration which we print the end of
    """
    log("-" * 40)
    log(f"END OF ITERATION {iteration}")
    log("-" * 40)

def log_config(msg):
    """
    Logs the config

    Args:
        msg: str
            The config that we print
    """
    log("\n")
    log("-" * 25 + "CONFIG" + "-" * 25)
    pp.pprint(f"{msg}")
    log("-" * 25 + "CONFIG" + "-" * 25 + "\n")

def setup_logger(filename='logfile.log', level=10, console_logging=True):
    """
    Setups logger

    Args:
        filename: str
            Name of file to log to
        level: int
            Logging level - levels: NOTSET=0, DEBUG=10, INFO=20, 
            WARNING=30, ERROR=40, CRITICAL=50
        console_logging: bool
            Flag to determine whether or not to log to console

    Returns:
        logger: logging instance
            Logger with desired config
    """
    # Creating log directory if it doesn't exist

    if "LOGDIR" in os.environ:
        logdir = os.environ["LOGDIR"]
    else:
        logdir = "logs"
    path = os.path.join(logdir, filename)
    # # Clear old logs
    # for f in os.listdir("logs"):
    #     os.remove(f)

    logger = logging.getLogger('ftml')
    # Setting level
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s() - %(levelname)s - %(message)s',
                                  "%Y-%m-%d %H:%M:%S")

    # FileHandler
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # StreamHandler which outputs to the console
    if console_logging:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def shut_down_logger(logger):
    logger.handlers = []
    logging.shutdown()

def convert_to_human_bytes(b):
    """
    Return the given bytes as a human friendly KB, MB, GB, or TB string

    Args:
        b: No. of bytes to be converted to human bytes

    Returns:
        human_bytes: str
            Converted string that is now human readable in either kilobytes, 
            megabytes, gigabytes or terabytes
    """
    b = float(b)
    kb = float(1024)
    mb = float(kb ** 2) # 1,048,576
    gb = float(kb ** 3) # 1,073,741,824
    tb = float(kb ** 4) # 1,099,511,627,776

    if b < kb:
        return '{0} {1}'.format(b,'Bytes' if 0 == b > 1 else 'Byte')
    elif kb <= b < mb:
        return '{0:.2f} KB'.format(b/kb)
    elif mb <= b < gb:
        return '{0:.2f} MB'.format(b/mb)
    elif gb <= b < tb:
        return '{0:.2f} GB'.format(b/gb)
    elif tb <= b:
        return '{0:.2f} TB'.format(b/tb)
