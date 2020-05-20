
"""Trainer class."""

import collections
import logging
import pathlib
import time

import matplotlib.pyplot as plt

import torch
from torch import optim
import tensorboardX as tb

import neuralprocess as npr


class Trainer:
    """Updater class for neural process.

    Args:
        model (neuralprocess.BaseNP): NP model.
        hparams (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: npr.BaseNP, hparams: dict):
        # Params
        self.model = model
        self.hparams = hparams

        # Attributes
        self.logdir = None
        self.logger = None
        self.writer = None
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

    def init_writer(self) -> None:
        """Initializes tensorboard writer."""

        self.writer = tb.SummaryWriter(str(self.logdir))

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

    def train(self, epoch: int) -> float:
        """Trains model.

        Args:
            epoch (int): Current epoch.

        Returns:
            train_loss (float): Accumulated loss per iteration.
        """

        # Logger for loss
        loss_dict = collections.defaultdict(float)

        # Run
        self.model.train()
        for data in self.train_loader:
            # Data to device
            data = (v.to(self.device) for v in data)

            # Forward
            self.optimizer.zero_grad()
            _tmp_loss_dict = self.model.loss_func(*data)
            loss = _tmp_loss_dict["loss"]

            # Backward
            loss.backward()
            self.optimizer.step()

            # Save loss
            for key, value in _tmp_loss_dict.items():
                loss_dict[key] += value.item()

        # Summary
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)

        return loss_dict["loss"]

    def test(self, epoch: int) -> float:
        """Tests model.

        Args:
            epoch (int): Current epoch.

        Returns:
            test_loss (float): Accumulated loss per iteration.
        """

        # Logger for loss
        loss_dict = collections.defaultdict(float)

        # Run
        self.model.eval()
        for data in self.test_loader:
            with torch.no_grad():
                # Data to device
                data = (v.to(self.device) for v in data)
                _tmp_loss_dict = self.model.loss_func(*data)

            # Save loss
            for key, value in _tmp_loss_dict.items():
                loss_dict[key] += value.item()

        # Summary
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"test/{key}", value, epoch)

        return loss_dict["loss"]

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

        self.writer.close()

    def save_plot(self, epoch: int) -> None:
        """Plot and save a figure.

        Args:
            epoch (int): Number of epoch.
        """

        # Query y_target
        x_ctx, y_ctx, x_tgt, y_tgt = next(iter(self.test_loader))
        with torch.no_grad():
            mu, logvar = self.model.query(x_ctx, y_ctx, x_tgt)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_tgt.squeeze(-1)[0], y_tgt.squeeze(-1)[0], "k:",
                 label="True function")
        plt.plot(x_ctx.squeeze(-1)[0], y_ctx.squeeze(-1)[0], "ko",
                 label="Context data")
        plt.plot(x_tgt.squeeze(-1)[0], mu.squeeze(-1)[0], "b",
                 label="Sampled function")
        plt.fill_between(
            x_tgt.squeeze(-1)[0],
            (mu + torch.exp(0.5 * logvar)).squeeze(-1)[0],
            (mu - torch.exp(0.5 * logvar)).squeeze(-1)[0],
            color="b", alpha=0.2, label="1-sigma range")
        plt.legend(loc="upper right")
        plt.title(f"Training results (epoch={epoch})")
        plt.tight_layout()
        plt.savefig(self.logdir / f"fig_{epoch}.png")
        plt.close()

    def _base_run(self) -> None:
        """Base running method."""

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
            self.train(epoch)

            if epoch % self.hparams["log_save_interval"] == 0:
                # Calculate test loss
                test_loss = self.test(epoch)

                # Save trained model
                self.save(epoch, test_loss)

                # Save plot
                self.save_plot(epoch)

        # Post process
        self.quit()

    def run(self) -> None:
        """Main run method."""

        # Settings
        self.check_logdir()
        self.init_logger()
        self.init_writer()

        self.logger.info("Start run")
        self.logger.info(f"Logdir: {self.logdir}")
        self.logger.info(f"Params: {self.hparams}")

        # Run
        try:
            self._base_run()
        except Exception as e:
            self.logger.exception(f"Run function error: {e}")

        self.logger.info("Finish run")
