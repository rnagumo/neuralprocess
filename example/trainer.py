
"""Trainer class."""

from typing import Tuple

import logging
import pathlib
import time

import torch
from torch import optim

import neuralprocess as npr


class Trainer:
    """Updater class for neural process.

    Args:
        model (neuralprocess.NeuralProcess): NP model.
        hparams (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model, hparams):
        # Params
        self.model = model
        self.hparams = hparams

        # Attributes
        self.logdir = None
        self.logger = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.device = None

    def check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        if "logdir" in self.hparams:
            logdir = pathlib.Path(self.hparams["logdir"])
        else:
            logdir = pathlib.Path("./logs/tmp/")

        self.logdir = logdir / time.strftime("%Y%m%d%H%M")
        self.logdir.mkdir(parents=True, exist_ok=True)

    def init_logger(self, save_file: bool = True) -> None:
        """Initalizes logger.

        Args:
            save_file (bool, optoinal): If `True`, save log file.
        """

        # Log file
        logpath = self.logdir / "training.log"

        # Initialize logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Set stream handler (console)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                   "- %(levelname)s : %(message)s")
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        # Set file handler (log file)
        if save_file:
            fh = logging.FileHandler(filename=logpath)
            fh.setLevel(logging.INFO)
            fh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                       "- %(levelname)s : %(message)s")
            fh.setFormatter(fh_fmt)
            logger.addHandler(fh)

        self.logger = logger

    def load_dataloader(self, train_dataset_params: dict,
                        test_dataset_params: dict) -> None:
        """Loads data loader for training and test.

        Args:
            train_dataset_params (dict): Dict of params for training dataset.
            test_dataset_params (dict): Dict of params for test dataset.
        """

        self.logger.info("Load dataset")

        self.train_loader = torch.utils.data.DataLoader(
            npr.GPDataset(train=True, **train_dataset_params),
            shuffle=True, batch_size=train_dataset_params["batch_size"])

        self.test_loader = torch.utils.data.DataLoader(
            npr.GPDataset(train=False, **test_dataset_params),
            shuffle=False, batch_size=test_dataset_params["batch_size"])

    def train(self) -> float:
        """Trains model.

        Returns:
            train_loss (float): Accumulated loss per iteration.
        """

        train_loss = 0.0
        self.model.train()
        for data in self.train_loader:
            # Data to device
            data = (v.to(self.device) for v in data)

            # Forward
            self.optimizer.zero_grad()
            loss_dict = self.model.loss_func(*data)
            loss = loss_dict.pop("loss")

            # Backward
            loss.backward()
            self.optimizer.step()

            # Save loss
            train_loss += loss.item()

        return train_loss

    def test(self) -> float:
        """Tests model.

        Returns:
            test_loss (float): Accumulated loss per iteration.
        """

        test_loss = 0.0
        self.model.eval()
        for data in self.test_loader:
            with torch.no_grad():
                # Data to device
                data = (v.to(self.device) for v in data)
                loss_dict = self.model.loss_func(*data)
            test_loss += loss_dict.pop("loss").item()

        return test_loss

    def save(self, epoch: int, loss: float) -> None:
        """Saves trained model and optimizer to checkpoint file.

        Args:
            epoch (int): Current epoch number.
            loss (float): Saved loss value.
        """

        # Log
        self.logger.info(f"Eval loss (epoch={epoch}): {loss}")
        self.logger.info("Save trained model")

        # Save model
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        path = self.logdir / f"checkpoint_{epoch}.pt"
        torch.save(state_dict, path)

    def quit(self) -> None:
        """Post process."""

        self.logger.info("Save data loader")
        torch.save(self.train_loader, self.logdir / "train_loader.pt")
        torch.save(self.test_loader, self.logdir / "test_loader.pt")

    def run(self) -> None:
        """Main run method."""

        # Settings
        self.check_logdir()
        self.init_logger()

        self.logger.info("Start run")
        self.logger.info(f"Logdir: {self.logdir}")
        self.logger.info(f"Args: {self.hparams}")

        # Data
        self.load_dataloader(self.hparams["train_dataset_params"],
                             self.hparams["test_dataset_params"])

        # Model to device
        if self.hparams["gpus"] is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.hparams['gpus']}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        for epoch in range(1, self.hparams["epochs"] + 1):
            # Training
            self.train()

            if epoch % self.hparams["log_save_interval"] == 0:
                # Calculate test loss
                test_loss = self.test()

                # Save trained model
                self.save(epoch, test_loss)

        # Post process
        self.quit()

        self.logger.info("Finish run")
