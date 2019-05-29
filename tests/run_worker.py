import click
import os
import json
import time

from fault_tolerant_ml.distribute import Worker

@click.command()
@click.argument('config_dir', type=click.Path(exists=True))
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--id', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
def run(config_dir, verbose, id, tmux):
    """Run worker

    Args:
        verbose (int): The debug level for the logging module
    """
    # load_dotenv(find_dotenv())

    identity: int = 0

    if tmux:
        identity = id=int(id[1:]) if id != "" else None
    else:
        identity = int(id) if id != "" else None

    worker = Worker(
        verbose=verbose,
        id=identity
    )

    time.sleep(1)

    with open(os.path.join(config_dir, "ip_config.json"), "r") as f:
        ip_config = json.load(f)

    ip_address = ip_config["ipAddress"]
    worker.connect(ip_address)
    # time.sleep(1)
    worker.start()

if __name__ == "__main__":
    run()