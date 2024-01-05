#!/bin/bash

# Switch on echoing
set -v

# Installing Java, Python and IPython
sudo apt update
sudo apt -y install openjdk-11-jre-headless openjdk-11-jdk python3 ipython3
sudo cp /usr/bin/python3 /usr/bin/python

# Installing pip for Python3
sudo apt install -y python3-pip

# Installing PySpark
pip3 install pyspark

# Getting Spark
mkdir -p ~/spark
if [ ! -f spark-3.1.2-bin-hadoop3.2.tgz ]; then
    wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
    tar -xvzf spark-3.1.2-bin-hadoop3.2.tgz -C ~/spark
fi

# Add lines to .bashrc
if [[ $(grep "SPARK" ~/.bashrc) ]]; then
    echo "Your .bashrc already has the needed lines."
else
    echo "export SPARK_HOME=~/spark/spark-3.1.2-bin-hadoop3.2" >> ~/.bashrc
    echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.bashrc
    source ~/.bashrc
fi

# Switch off echoing
set +v

# Done
echo "Enjoy using Spark!"
