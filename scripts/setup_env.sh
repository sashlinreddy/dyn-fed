#!/bin/bash

# ssh-keygen -t rsa -f ~/.ssh/github
# Enter passphrase
# Add key to github

# git clone git@github.com:sashlinreddy/fault-tolerant-ml.git

# Install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

echo "# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home-mscluster/sreddy/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval \"$__conda_setup\"
else
    if [ -f \"/home-mscluster/sreddy/miniconda3/etc/profile.d/conda.sh\" ]; then
        . \"/home-mscluster/sreddy/miniconda3/etc/profile.d/conda.sh\"
    else
        export PATH=\"/home-mscluster/sreddy/miniconda3/bin:$PATH\"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<" >> ~/.bashrc

# Create environment
conda create -n ftml python=3.6 -y

# cd ~/fault-tolerant-ml
# pip install -r requirements_dev.txt

# Install protobuf
mkdir protoc-3.10.1 && cd protoc-3.10.1
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protoc-3.10.1-linux-x86_64.zip
unzip protoc-3.10.1-linux-x86_64.zip

echo 'export PROTOBUF_HOME=$HOME/protoc-3.10.1/' >> ~/.bashrc
echo 'export PATH=${PROTOBUF_HOME}/bin:${PATH}' >> ~/.bashrc

# scp gcp
# gcloud compute scp data/mnist/*.gz g1-login1:/home/g675723_students_wits_ac_za/fault-tolerant-ml/data/mnist
# gcloud compute scp data/fashion-mnist/*.gz g1-login1:/home/g675723_students_wits_ac_za/fault-tolerant-ml/data/fashion-mnist
