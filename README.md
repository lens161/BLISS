# BLISS
Implementation of the BLISS algorithm for our Master Thesis: 
"Implement and improve “BLISS”: 
Optimising a Learned Index for 
Approximate Nearest Neighbour Search on Commodity Hardware"

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, scikit-learn, pandas, h5py, psutil, faiss

conda.yaml can be used to install all requirements at once on an Anaconda environment

##  Quick setup

First choose a dataset:
dataset_name: e.g. "sift-128-euclidean", "glove-100-angular", etc. Many datasets are already supported. For more datasets to be supported add an additional custom dataset objects.

All hyper-parameters live in config.py’s Config class and have defaults that should be optimal for most cases. They can be overridden for experimentation. Most 1B datasets require axel or azcopy to be downloded. 

However always override the following:
- datasize -> should be set to the 1 for 1M Datasets or to 10, 100 or 1000 for 10M, 100M and 1B respectively.

Simply create a config object (or multiple) with the desired hyper-parameters and run it/them with the provided build and query functions from experiments.py. Or design your own experimental pipeline if required.

## Acknowledgements

This Project is inspired by the paper "BLISS: A Billion scale Index using Iterative Re-partitioning" and the corresponding implementation by the author: https://github.com/gaurav16gupta/BLISS

Code for dataset management: https://github.com/harsha-simhadri/big-ann-benchmarks

### Created by
Nikki Allegonda Maria Van Gasteren, Lennart Arne Sack
