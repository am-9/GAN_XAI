#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=62000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=72:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

module load python/3.6
virtualenv --no-download /home/amahalan/scratch/env
source /home/amahalan/scratch/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch
pip install -vvv captum
pip install certifi
pip install cycler
pip install future
pip install kiwisolver
pip install matplotlib
pip install numpy
pip install Pillow
pip install protobuf
pip install pyparsing
pip install python-dateutil
pip install six
pip install tensorboardX
pip install torchvision
pip install fastdtw
pip install wfdb

cd /home/amahalan/scratch/GAN_XAI/src
python main.py

