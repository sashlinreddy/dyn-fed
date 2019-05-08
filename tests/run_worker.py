import click
import os
import json

from fault_tolerant_ml.distribute import Worker

@click.command()
@click.argument('config_dir', type=click.Path(exists=True))
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--id', '-i', default="", type=str)
def run(config_dir, verbose, id):
    """Run worker

    Args:
        verbose (int): The debug level for the logging module
    """
    # load_dotenv(find_dotenv())

    worker = Worker(
        verbose=verbose,
        id=int(id[1:]) if id != "" else None
    )


    with open(os.path.join(config_dir, "ip_config.json"), "r") as f:
        ip_config = json.load(f)

    ip_address = ip_config["ipAddress"]
    worker.connect(ip_address)
    # time.sleep(1)
    worker.start()

if __name__ == "__main__":
    run()