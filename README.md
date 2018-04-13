# GraphLSTM for Hand Pose Estimation
This repository implements a tensorflow library for hand pose estimation. The network architecture is based on [GraphLSTM (Liang et al.)](https://arxiv.org/abs/1603.07063)
## Getting started
*GraphLSTM* extends the *Tensorflow* framework and is written in Python 3.

 In order to run the provided test code or experiment with your own, a few requirements have to be met.
### Clone the Repository
Get the code by cloning the repository into your directory of choice:

    git clone https://github.com/mqne/GraphLSTM.git
and `cd` into it.

    cd GraphLSTM

### Prerequisites
Apart from *[Tensorflow](https://www.tensorflow.org/install/)*, the following dependencies need to be fulfilled in order to get *GraphLSTM* up and running:

 - `python 3.6` or greater
 - `networkx 2.0` or greater

You also need `numpy` which is automatically installed with Tensorflow.

*This is enough to run the core package.* The following packages are required for additional functionality:

 - `matplotlib` for plotting the *GraphLSTM* graph,
 - `pandas` and
 - `scipy` for checks executed within the `networkx` library.

#### Installing via Anaconda

For creating a new [Anaconda](https://www.anaconda.com/download/) environment *graphlstm* with all necessary and recommended packages installed, execute the following inside your repo directory:

*If you have a GPU with support for *CUDA*:*

	conda create -n graphlstm --file requirements.txt python=3.6
*If you do NOT have a GPU with *CUDA* support:*
 
Edit the file `requirements.txt` and replace the line `tensorflow-gpu` with `tensorflow`, then run above code.

## Running the provided test code
Make sure to have your python environment activated (i.e. for an *Anaconda* installation, run `source activate graphlstm` ).
Then, simply run:

    python unit_tests.py
  
## Authors
- Matthias Kühne - [mqne](https://www.github.com/mqne)
