
"""Training method."""

import argparse
import json
import os
import pathlib
import random

import torch

import neuralprocess as npr
from experiment import Trainer


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Command line args
    args = init_args()

    # Configs
    config_path = pathlib.Path(
        os.getenv("CONFIG_PATH", "./examples/config_1d.json"))
    with config_path.open() as f:
        config = json.load(f)

    # Path
    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"),
                              os.getenv("EXPERIMENT_NAME", "tmp")))

    # Cuda setting
    use_cuda = torch.cuda.is_available() and args.cuda != "null"
    gpus = args.cuda if use_cuda else None

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------

    # NP model
    model_dict = {
        "cnp": npr.ConditionalNP,
        "np": npr.NeuralProcess,
        "anp": npr.AttentiveNP,
    }
    model = model_dict[args.model](**config[f"{args.model}_params"])

    # Trainer
    params = {
        "logdir": logdir,
        "gpus": gpus,
    }
    params.update(config)
    params.update(vars(args))

    trainer = Trainer(model, params)
    trainer.run()


def init_args():
    parser = argparse.ArgumentParser(description="NP training")
    parser.add_argument("--model", type=str, default="cnp",
                        help="Trained model name.")
    parser.add_argument("--cuda", type=str, default="0",
                        help="Number of CUDA device. 'null' means cpu device.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--log-save-interval", type=int, default=10,
                        help="Interval epochs of saving logs.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
