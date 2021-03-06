
"""Trainer class."""

from typing import DefaultDict, Optional

import collections
import dataclasses
import json
import logging
import pathlib
import time

import matplotlib.pyplot as plt
import tqdm

import torch
from torch import optim, Tensor
import tensorboardX as tb

import neuralprocess as npr


@dataclasses.dataclass
class Config:
    # From kwargs
    model: str
    cuda: str
    seed: int
    max_epochs: int
    log_interval: int

    # From config file
    train_dataset_params: dict
    test_dataset_params: dict
    cnp_params: dict
    np_params: dict
    dnp_params: dict
    anp_params: dict
    convcnp_params: dict
    fnp_params: dict
    snp_params: dict

    # From params
    logdir: str
    gpus: Optional[str]


class Trainer:
    """Trainer class for neural process.

    Args:
        model (neuralprocess.BaseNP): NP model.
        config (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: npr.BaseNP, config: dict) -> None:
        # Params
        self.model = model
        self.config = Config(**config)

        # Attributes
        self.logdir = pathlib.Path()
        self.logger = logging.Logger("")
        self.writer: tb.SummaryWriter
        self.train_loader: torch.utils.data.dataloader.DataLoader
        self.test_loader: torch.utils.data.dataloader.DataLoader
        self.optimizer: optim.optimizer.Optimizer
        self.device: torch.device
        self.epoch = 0

    def check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        self.logdir = pathlib.Path(
            self.config.logdir, time.strftime("%Y%m%d%H%M"))
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
            fh.setLevel(logging.DEBUG)
            fh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                       "- %(levelname)s : %(message)s")
            fh.setFormatter(fh_fmt)
            logger.addHandler(fh)

        self.logger = logger

    def init_writer(self) -> None:
        """Initializes tensorboard writer."""

        self.writer = tb.SummaryWriter(str(self.logdir))

    def load_dataloader(self) -> None:
        """Loads data loader for training and test."""

        self.logger.info("Load dataset")

        if self.config.model == "snp":
            self.train_loader = torch.utils.data.DataLoader(
                npr.SequentialGPDataset(
                    train=True, seq_len=20,
                    **self.config.train_dataset_params),
                shuffle=True, batch_size=16)

            self.test_loader = torch.utils.data.DataLoader(
                npr.SequentialGPDataset(
                    train=False, seq_len=20,
                    **self.config.test_dataset_params),
                shuffle=False, batch_size=1)
        else:
            self.train_loader = torch.utils.data.DataLoader(
                npr.GPDataset(train=True, **self.config.train_dataset_params),
                shuffle=True, batch_size=16)

            self.test_loader = torch.utils.data.DataLoader(
                npr.GPDataset(train=False, **self.config.test_dataset_params),
                shuffle=False, batch_size=1)

        self.logger.info(f"Train dataset size: {len(self.train_loader)}")
        self.logger.info(f"Test dataset size: {len(self.test_loader)}")

    def train(self) -> float:
        """Trains model.

        Returns:
            train_loss (float): Accumulated loss per iteration.
        """

        # Logger for loss
        loss_dict: DefaultDict[str, float] = collections.defaultdict(float)

        # Resample dataset with/without kernel hyper-parameter update
        resample_params = self.config.model in ("dnp", "anp")
        self.train_loader.dataset.generate_dataset(
            resample_params=resample_params, single_params=False)

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
            self.writer.add_scalar(f"train/{key}", value, self.epoch)

        return loss_dict["loss"]

    def test(self) -> float:
        """Tests model.

        Returns:
            test_loss (float): Accumulated loss per iteration.
        """

        # Logger for loss
        loss_dict: DefaultDict[str, float] = collections.defaultdict(float)

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
            self.writer.add_scalar(f"test/{key}", value, self.epoch)

        return loss_dict["loss"]

    def save_checkpoint(self) -> None:
        """Saves trained model and optimizer to checkpoint file."""

        # Log
        self.logger.debug("Save trained model")

        # Remove unnecessary prefix from state dict keys
        model_state_dict = {}
        for k, v in self.model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self.optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        # Save model
        state_dict = {
            "epoch": self.epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }

        path = self.logdir / f"checkpoint_{self.epoch}.pt"
        torch.save(state_dict, path)

    def save_configs(self) -> None:
        """Saves setting including condig and args in json format."""

        config = dataclasses.asdict(self.config)
        config["logdir"] = str(self.logdir)

        with (self.logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def save_plot(self) -> None:
        """Plot and save a figure."""

        # Sample y_target
        x_ctx, y_ctx, x_tgt, y_tgt = next(iter(self.test_loader))
        with torch.no_grad():
            data = (x_ctx.to(self.device), y_ctx.to(self.device),
                    x_tgt.to(self.device))
            mu, var = self.model.sample(*data)

        mu = mu.cpu()
        var = var.cpu()

        if self.config.model == "snp":
            self._plot_sequence(x_ctx, y_ctx, x_tgt, y_tgt, mu, var)
        else:
            self._plot_single(x_ctx, y_ctx, x_tgt, y_tgt, mu, var)

    def _plot_single(self, x_ctx: Tensor, y_ctx: Tensor, x_tgt: Tensor,
                     y_tgt: Tensor, mu: Tensor, var: Tensor) -> None:
        """Plots single figure.

        Args:
            x_ctx (torch.Tensor): Context x, size `(batch, num_ctx, 1)`.
            y_ctx (torch.Tensor): Context y, size `(batch, num_ctx, 1)`.
            x_tgt (torch.Tensor): Target x, size `(batch, num_tgt, 1)`.
            y_tgt (torch.Tensor): Target y, size `(batch, num_tgt, 1)`.
            mu (torch.Tensor): Sampled mu, size `(batch, num_tgt, 1)`.
            var (torch.Tensor): Sampled var, size `(batch, num_tgt, 1)`.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(x_tgt.squeeze(-1)[0], y_tgt.squeeze(-1)[0], "k:",
                 label="True function")
        plt.plot(x_ctx.squeeze(-1)[0], y_ctx.squeeze(-1)[0], "ko",
                 label="Context data")
        plt.plot(x_tgt.squeeze(-1)[0], mu.squeeze(-1)[0], "b",
                 label="Sampled function")
        plt.fill_between(x_tgt.squeeze(-1)[0],
                         (mu + var ** 0.5).squeeze(-1)[0],
                         (mu - var ** 0.5).squeeze(-1)[0],
                         color="b", alpha=0.2, label="1-sigma range")
        plt.legend(loc="upper right")
        plt.title(f"Training results (epoch={self.epoch})")
        plt.tight_layout()
        plt.savefig(self.logdir / f"fig_{self.epoch}.png")
        plt.close()

    def _plot_sequence(self, x_ctx: Tensor, y_ctx: Tensor, x_tgt: Tensor,
                       y_tgt: Tensor, mu: Tensor, var: Tensor,
                       skip_step: int = 4) -> None:
        """Plots single figure.

        Args:
            x_ctx (torch.Tensor): Context x, size `(b, l, n, 1)`.
            y_ctx (torch.Tensor): Context y, size `(b, l, n, 1)`.
            x_tgt (torch.Tensor): Target x, size `(b, l, m, 1)`.
            y_tgt (torch.Tensor): Target y, size `(b, l, m, 1)`.
            mu (torch.Tensor): Sampled mu, size `(b, l, m, 1)`.
            var (torch.Tensor): Sampled var, size `(b, l, m, 1)`.
            skip_step (int, optional): Skip length to plot.
        """

        seq_len = x_ctx.size(1)
        total = seq_len // skip_step

        plt.figure(figsize=(10, 24))
        for i, t in enumerate(range(0, seq_len, skip_step)):
            plt.subplot(total, 1, i + 1)
            plt.plot(x_tgt.squeeze(-1)[0, t], y_tgt.squeeze(-1)[0, t], "k:",
                     label="True function")
            plt.plot(x_ctx.squeeze(-1)[0, t], y_ctx.squeeze(-1)[0, t], "ko",
                     label="Context data")
            plt.plot(x_tgt.squeeze(-1)[0, t], mu.squeeze(-1)[0, t], "b",
                     label="Sampled function")
            plt.fill_between(x_tgt.squeeze(-1)[0, t],
                             (mu + var ** 0.5).squeeze(-1)[0, t],
                             (mu - var ** 0.5).squeeze(-1)[0, t],
                             color="b", alpha=0.2, label="1-sigma range")
            plt.legend(loc="upper right")
            plt.title(f"Time step {t}")
            plt.tight_layout()

        plt.savefig(self.logdir / f"fig_{self.epoch}.png")
        plt.close()

    def quit(self) -> None:
        """Post process."""

        self.logger.info("Save data loader")
        torch.save(self.train_loader, self.logdir / "train_loader.pt")
        torch.save(self.test_loader, self.logdir / "test_loader.pt")

        self.save_configs()
        self.writer.close()

    def _base_run(self) -> None:
        """Base running method."""

        self.logger.info("Start experiment")

        # Data
        self.load_dataloader()

        # Model to device
        if self.config.gpus is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.config.gpus}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # Run training
        self.logger.info("Start training")

        pbar = tqdm.trange(1, self.config.max_epochs + 1)
        postfix = {"train/loss": 0.0, "test/loss": 0.0}
        self.epoch = 0

        for _ in pbar:
            self.epoch += 1

            # Training
            train_loss = self.train()
            postfix["train/loss"] = train_loss

            if self.epoch % self.config.log_interval == 0:
                # Calculate test loss
                test_loss = self.test()
                postfix["test/loss"] = test_loss
                self.save_checkpoint()
                self.save_plot()

            # Update postfix
            pbar.set_postfix(postfix)

    def run(self) -> None:
        """Main run method."""

        # Settings
        self.check_logdir()
        self.init_logger()
        self.init_writer()

        self.logger.info("Start run")
        self.logger.info(f"Logdir: {self.logdir}")
        self.logger.info(f"Params: {self.config}")

        # Run
        try:
            self._base_run()
        except Exception as e:
            self.logger.exception(f"Run function error: {e}")
        finally:
            self.quit()

        self.logger.info("Finish run")
