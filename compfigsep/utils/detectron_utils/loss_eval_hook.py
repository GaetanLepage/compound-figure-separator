"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.fr
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


####################################################
LossEvalHook to evaluate loss on the validation set.
"""

import time
import datetime
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger


class LossEvalHook(HookBase):
    """
    Custom Hook (subclassing detectron2's `HookBase`) to evaluate loss on the validation set.
    """

    def __init__(self,
                 eval_period: int,
                 model: nn.Module,
                 data_loader: DataLoader
                 ) -> None:
        """
        Init function.

        Args:
            eval_period (int):          The period (number of steps) to run the evaluation on the
                                            validation set.
            model (nn.Module):          The model that the validation set will be evaluated on.
            data_loader (DataLoader):   The DataLoader yielding samples from the evaluation data
                                            set.
        """
        self._model: nn.Module = model
        self._data_loader: DataLoader = data_loader
        self._period: int = eval_period
        self._logger: logging.Logger = setup_logger(name=__name__,
                                                    distributed_rank=comm.get_rank())

    def _do_loss_eval(self) -> float:
        """
        Evaluate the loss function on the validation set.

        Returns:
            mean_loss (float):  Value of the loss.
        """
        # Copying inference_on_dataset from evaluator.py
        num_samples: int = len(self._data_loader)
        self._logger.info("Starting validation on %d samples",
                          num_samples)
        num_warmup: int = min(5, num_samples - 1)

        start_time: float = time.perf_counter()
        total_compute_time: float = 0
        losses: list[float] = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            # Inference for these inputs
            start_compute_time: float = time.perf_counter()
            loss_batch: float = self._get_loss(inputs)
            losses.append(loss_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)

            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                # Compute average time spent on each image.
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start

                # Compute ETA
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (num_samples - idx - 1)))

                log_every_n_seconds(lvl=logging.INFO,
                                    msg=f"Loss on Validation done {idx + 1}/{num_samples}."
                                        f" {seconds_per_img:.4f} s / img. ETA={eta}",
                                    n=100,
                                    name=__name__)

        # Average the losses.
        mean_loss = np.mean(losses)

        # Print the loss value.
        self._logger.info("Validation loss : {mean_loss}")

        # Store the loss value for it to be logged and displayed in TensorBoard.
        self.trainer.storage.put_scalar('validation_loss',
                                        mean_loss)
        comm.synchronize()

        return mean_loss

    def _get_loss(self, data: dict) -> float:
        """
        Compute the loss value for a single sample.

        Args:
            data (dict): A sample from the validation set.

        Returns:
            total_losses_reduced (float): Loss value
        """
        loss_dict = self._model(data)
        loss_dict = {
            k: v.detach().cpu().item()
            if isinstance(v, torch.Tensor)
            else float(v)
            for k, v in loss_dict.items()
        }
        total_losses_reduced = sum(loss_dict.values())
        return total_losses_reduced

    def after_step(self):
        """
        Evaluate the loss function on the validation set at the end of the training process.
        """
        next_iter: int = self.trainer.iter + 1
        is_final: bool = next_iter == self.trainer.max_iter

        # If training is over:
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            # Evaluate loss on validation set.
            self._do_loss_eval()
