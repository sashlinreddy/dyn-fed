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
pythonw tests/run_master.py $n_workers # Run in separate window
pythonw tests/run_worker.py 7 -i $TMUX_PANE -t 1 # Set "setw synchronize-panes on" as a tmux setting. Use Ctrl+B,: for insert mode
```

To view the results on tensorboard assuming you are in the parent directory:

```bash
tensorboard --logdir=logs
```

Go to http://localhost:6006.

## Running on SLURM cluster

```bash
sbatch -n $ntasks fault-tolerant-ml/slurm_launch.sh
```

The [slurm launch](slurm_launch.sh) generates a multi-prog on the fly with desired arguments. The above command will launch a job with the default arguments specified in [master execution script](tests/run_master.py). However, arguments can be passed to the job submission as below:

```bash
sbatch -n $ntasks fault-tolerant-ml/slurm_launch.sh -s 8 -t 25 -v 20 -cp 2 -q 1
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
