import click

from fault_tolerant_ml.distribute import Worker

@click.command()
@click.option('--verbose', '-v', default=10, type=int)
def run(verbose):
    """Run worker

    Args:
        verbose (int): The debug level for the logging module
    """
    # load_dotenv(find_dotenv())

    worker = Worker(
        verbose=verbose
    )
    worker.connect()
    # time.sleep(1)
    worker.start()

if __name__ == "__main__":
    run()