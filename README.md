# Fault Tolerant Machine Learning

Fault tolerant framework for machine learning algorithms

____

* Free software: MIT license
* Documentation: https://fault-tolerant-ml.readthedocs.io.

## Prerequisites

* python >= 3.6
* [protocol buffers](https://github.com/protocolbuffers/protobuf/releases)

## Datasets

* Occupancy data (https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)
* MNIST (http://yann.lecun.com/exdb/mnist/)
* Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist)

## Features

* Parameter quantization
* Fault tolerant training - master keeps track of workers most representative data samples. If the worker goes down, the master redistributes those points to the workers that continue to do work
* Periodic communication - Workers only communicate parameters based on a selected interval

## Development

```bash
git clone https://github.com/sashlinreddy/fault-tolerant-ml.git
```

First you need to compile the protocol buffer file. The definitions are in the [ftml.proto](protos/ftml.proto) file.

Compilation is executed with the following command:

```bash
protoc -I=protos/ --python_out=fault_tolerant_ml/proto/ protos/ftml.proto
```

### Local development (With tmux)

```bash
tmux
export LOGDIR=${PWD}/logs
pythonw tests/run_master.py $n_workers # Run in separate window
pythonw tests/run_worker.py $n_workers -i $TMUX_PANE -t 1 # Set "setw synchronize-panes on" as a tmux setting. Use Ctrl+B,: for insert mode
```

To view the results on tensorboard assuming you are in the parent directory:

```bash
tensorboard --logdir=logs
```

Go to http://localhost:6006.

### Running on SLURM cluster

```bash
sbatch -n $ntasks fault-tolerant-ml/slurm_launch.sh
```

The [slurm launch](slurm_launch.sh) generates a multi-prog on the fly with desired arguments. The above command will launch a job with the default arguments specified in [master execution script](scripts/run_master.py). However, arguments can be passed to the job submission as below:

```bash
sbatch -n $ntasks fault-tolerant-ml/slurm_launch.sh -v 20
```

## Setup config

The config of the model can be set in the [config file](config.yml). The dataset can be configured in this file as well as the following parameters:

* model
    * n_iterations:  No. of iterations
    * shuffle: Whether or not to shuffle the data in each iteration

* optimizer
    * learning_rate: Rate at which model learns
        * Mnist: SGD: 0.99, Adam: 0.01
        * Fashion Mnist: SGD: 0.25, Adam: 0.001
    * mu_g: Weighting given to global W when workers updating local parameters. 0.0 for normal local update.
    * n_most_rep: No. of most representative data points to keep track of when worker goes down
    * name: Name of optimizer (Currently supports sgd and adam)

* executor:
    * strategy: Name of distribution strategy
    * scenario: Scenario type, see code for more details.
    * remap: Redistribution strategy
    * quantize: Whether or not to use quantization when communicating parameters
    * comm_period: How often to communicate parameters
    * delta_switch: When to switch to every iteration communication
    * timeout: Time given for any workers to join
    * send_gradients: Whether or not to send gradients back to master
    * shared_folder: Dataset to be used

## View Results

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

To pull results for log files run the following:

```bash
python fault-tolerant-ml/fault_tolerant_ml/utils/logger_parser.py logs/slurm/[fashion-mnist|mnist]/ fault-tolerant-ml/data/[fashion_mnist|mnist]_results.csv
```

## Run tests

```bash
nosetests -vv
```

## Future

* Build and test on mobile app

## Links

* [Automatic differentiation lecture notes](http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf)
* [Automatic differentation wiki](https://en.wikipedia.org/wiki/Automatic_differentiation)
* https://github.com/joelgrus/autograd/tree/master
* https://github.com/ddbourgin

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
