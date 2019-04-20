# Fault Tolerant Machine Learning

Fault tolerant framework for machine learning algorithms

* Free software: MIT license
* Documentation: https://fault-tolerant-ml.readthedocs.io.

## Features

* TODO

## Development
```bash
export LOGDIR=${PWD}/logs
./run.sh
python tests/run_master.py
python tests/run_worker.py -i ${TMUX_PANE}
```
## Run tests

```bash
nosetests --exe
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
