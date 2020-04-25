# Dynamic Averaging techniques for Federated Machine Learning

Dynamic averaging framework for federated machine learning algorithms

____

* Free software: MIT license

## Prerequisites

* python >= 3.6
* [protocol buffers](https://github.com/protocolbuffers/protobuf/releases)

## Datasets

* Occupancy data (https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)
* MNIST (http://yann.lecun.com/exdb/mnist/)
* Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist)

## Features

* Parameter quantization
* Fault tolerant training - server keeps track of clients most representative data samples. If the worker goes down, the server redistributes those points to the clients that continue to do work
* Periodic communication - Workers only communicate parameters based on a selected interval

## Development

```bash
git clone https://github.com/sashlinreddy/dyn-fed.git
```

First you need to compile the protocol buffer file. The definitions are in the [dfl.proto](protos/dfl.proto) file.

Compilation is executed with the following command:

```bash
protoc -I=protos/ --python_out=dyn_fed/proto/ protos/dfl.proto
```

### Local development (With tmux)

```bash
tmux
export LOGDIR=${PWD}/logs
./scripts/client_local.sh $N_WORKERS -v $VERBOSE -m $MODEL_TYPE # Run in separate window
./scripts/server_local.sh $N_WORKERS -v $VERBOSE -m $MODEL_TYPE # Set "setw synchronize-panes on" as a tmux setting. Use Ctrl+B,: for insert mode
```

To view the results on tensorboard assuming you are in the parent directory:

```bash
tensorboard --logdir=logs
```

Go to http://localhost:6006.

### Running on SLURM cluster

The [slurm launch](scripts/slurm_launch.sh) generates a multi-prog on the fly with desired arguments. The above command will launch a job with the default arguments specified in [server execution script](examples/tf_train_model.py). However, arguments can be passed to the job submission as below:

```bash
sbatch -n $ntasks dyn-fed/scripts/slurm_launch.sh -m $MODEL_TYPE -v $VERBOSE
```

## Generate multiple experiments

```bash
(ftml) $ python dyn-fed/examples/train_experiments.py
```

## Cancel multiple jobs

```bash
squeue -u $USER | grep $JOBIDSTART |awk '{print $1}' | xargs -n 1 scancel
```

## Setup config

The config of the model can be set in the [config file](config/config.yml). The dataset can be configured in this file as well as the following parameters:

* model
    * type: Type of model
    * n_iterations:  No. of iterations
    * shuffle: Whether or not to shuffle the data in each iteration

* data
  * name: Dataset name
  * shuffle: Whether or not to shuffle dataset # Not used
  * batch_size: Data batch size
  * shuffle_buffer_size: Shuffle buffer size
  * noniid: Whether or not to make dataset noniid
  * unbalanced: Whether or not to make dataset unbalanced

* optimizer
    * learning_rate: Rate at which model learns
        * Mnist: SGD: 0.01, Adam: 0.001
        * Fashion Mnist: SGD: 0.01, Adam: 0.001
    * name: Name of optimizer (Currently supports sgd and adam)

* distribute
    * strategy: Name of distribution strategy
    * remap: Redistribution strategy
    * quantize: Whether or not to use quantization when communicating parameters
    * comm_period: How often to communicate parameters
    * delta_switch: When to switch to every iteration communication
    * delta_threshold: For dynamic averaging paper
    * timeout: Time given for any clients to join
    * send_gradients: Whether or not to send gradients back to server
    * shared_folder: Dataset to be used

* executor
    * scenario: Scenario type, see code for more details

## View Results

To view the results on tensorboard:

```bash
sbatch scripts/tensorboard_slurm.sh
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
* https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
* https://github.com/joelgrus/autograd/tree/master
* https://github.com/ddbourgin

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
