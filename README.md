# Fault Tolerant Machine Learning

Fault tolerant framework for machine learning algorithms

* Free software: MIT license
* Documentation: https://fault-tolerant-ml.readthedocs.io.

## Datasets

* Occupancy data (https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)
* MNIST (http://yann.lecun.com/exdb/mnist/)
* Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist)

## Features

* Parameter quantization
* Fault tolerant training - master keeps track of workers most representative data samples. If the worker goes down, the master redistributes those points to the workers that continue to do work
* Periodic communication - Workers only communicate parameters based on a selected interval

## Local development (With tmux)

```bash
tmux
export LOGDIR=${PWD}/logs
python tests/run_master.py # Run in separate window
python tests/run_worker.py -i ${TMUX_PANE} # Set "setw synchronize-panes on" as a tmux setting. Use Ctrl+B,: for insert mode
```

To view the results on tensorboard assuming you are in the parent directory:

```bash
tensorboard --logdir=logs
```

Go to http://localhost:6006.

## Running on SLURM cluster

```bash
sbatch slurm_launch.sh
```

To view the results on tensorboard:

```bash
sbatch tensorboard_slurm.sh
```

Check the output log located in $HOME/logs. 

We need to create an ssh tunnel 
```bash
ssh username@$clusterIP -L $localPort:$clusterNodeip:$clusterNodePort
```

View tensorboard on http://localhost:6006 :)

## Run tests

```bash
nosetests --exe
```

## Future

* Build and test on mobile app

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
