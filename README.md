
# neuralprocess

Neural Process family

# Requirements

* Python == 3.7
* PyTorch == 1.5.0

Requirements for example code

* numpy == 1.18.4
* matplotlib == 3.2.1
* tqdm == 4.46.0
* tensorboardX == 2.0

# How to use

## Set up environments

Clone repository.

```bash
git clone https://github.com/rnagumo/neuralprocess.git
cd neuralprocess
```

Install the package in virtual env.

```bash
python -m venv .venv
source .venv/bin/activate
pip install .

# Install other requirements.
pip install numpy==1.18.4 matplotlib==3.2.1 tqdm==4.46.0 tensorboardX==2.0
```

Or use [Docker](https://docs.docker.com/get-docker/).

```bash
docker build -t neuralprocess .
docker run -it neuralprocess bash
```

## Run experiment

Train models. Shell scripts in `bin` folder contains the necessary settings for building the environment.

```bash
# Usage
bash bin/train.sh <model-name> <random-seed>

# Example
bash bin/train.sh cnp 0
```

# Reference

* DeepMind "The Neural Process Family" ([GitHub](https://github.com/deepmind/neural-processes))
* cambridge-mlg "Convolutional Conditional Neural Processes" ([GitHub](https://github.com/cambridge-mlg/convcnp))
