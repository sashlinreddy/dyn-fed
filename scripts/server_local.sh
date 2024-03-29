#!/bin/bash
NWORKERS=$1
echo "No of clients: $NWORKERS"

while [ "$2" != "" ]; do
    case $2 in
        -v| --VERBOSE )         shift
                                VERBOSE="-v $2"
                                ;;
        -m| --MODEL )         shift
                                MODEL="$2"
                                ;;
        -c| --config )          shift
                                config="-c $2"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Decide on the type of model
if [ "$MODEL" == "LINEAR" ]; then
    echo "Running Linear Model"
    pythonw examples/train_linear.py $NWORKERS -r server $VERBOSE
elif [ "$MODEL" == "LOGISTIC" ]; then
    echo "Running Logistic Model"
    pythonw examples/train_logistic.py $NWORKERS -r server $VERBOSE
elif [ "$MODEL" == "V2" ]; then
    echo "Running Version2 Model"
    pythonw examples/train_model.py $NWORKERS -r server $VERBOSE $config
elif [ "$MODEL" == "NN" ]; then
    echo "Running NN Model"
    pythonw examples/train_nn.py $NWORKERS -r server $VERBOSE
elif [ "$MODEL" == "TF" ]; then
    echo "Running TF Model"
    pythonw examples/train_tf_model.py $NWORKERS -r server $VERBOSE $config
else
    echo "Running Version2 Model"
    pythonw examples/train_model.py $NWORKERS -r server $VERBOSE $config
fi