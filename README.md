# Graph LSTM for Hand Pose Estimation
This repository implements a Tensorflow library for hand pose estimation. The architecture of the refinement stage is based on the [2016 CVPR paper](https://arxiv.org/abs/1603.07063) by Liang et al.

![Hand Pose Estimation Example Image](https://github.com/mqne/GraphLSTM/raw/master/example_pair.png)

## Getting started
*Graph LSTM* extends the *Tensorflow* framework and is written in Python 3.

 In order to run the provided test code or experiment with your own, a few requirements have to be met.
### Clone the Repository
Get the code by cloning the repository into your directory of choice:

    git clone https://github.com/mqne/GraphLSTM.git
and `cd` into it.

    cd GraphLSTM

### Prerequisites
 - `tensorflow 1.8` or greater
 
Apart from *[Tensorflow](https://www.tensorflow.org/install/)*, the following dependencies need to be fulfilled in order to get *Graph LSTM* up and running:

 - `python 3.6` or greater
 - `networkx 2.0` or greater

You also need `numpy` which is automatically installed with Tensorflow.

*This is enough to run the core package.* The following packages are required for additional functionality:

 - `matplotlib` for plotting the *Graph LSTM* graph and network evaluation plots,
 - `pandas` and
 - `scipy` for checks executed within the `networkx` library.

For running the *Region Ensemble network*, these packages are additionally required:

- `keras`
- `seaborn`
- `plotly`
- `h5py`
- `scikit-learn`
- `tqdm`
- `pydot`
- `pillow`

#### Installing via Anaconda
For creating a new [Anaconda](https://www.anaconda.com/download/) environment *graphlstm* with all necessary and recommended packages installed, do the following:

- *Only if you have a GPU with support for CUDA:* Edit the file `requirements.txt` and replace the line `tensorflow` with `tensorflow-gpu` as well as the line `keras` with `keras-gpu`.

Run the following inside your repo directory:

	conda create -n graphlstm --file requirements.txt python=3.6

As `opencv` for Python 3.6 is not available in the default anaconda repositories, you have to install it manually via:

	conda install -n graphlstm -c menpo opencv3

## Running the provided test code
Make sure to have your python environment activated (i.e. for an *Anaconda* installation, run `source activate graphlstm` ).
Then, simply run:

    python unit_tests.py
  
## Authors
- Matthias Kühne - [mqne](https://www.github.com/mqne)
