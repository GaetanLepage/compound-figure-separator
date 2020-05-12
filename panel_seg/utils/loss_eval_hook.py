"""
TODO doc
"""

import time
import datetime
import logging
from collections import OrderedDict

import torch
import numpy as np


from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger


class LossEvalHook(HookBase):
    """
    TODO doc
    """

    def __init__(self, eval_period, model, data_loader):
        """
        TODO
        """
        self._model = model
        self._data_loader = data_loader
        self._period = eval_period
        self._logger = setup_logger(name=__name__,
                                    distributed_rank=comm.get_rank())

    def _do_loss_eval(self):
        """
        TODO
        """
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        self._logger.info("Starting validation on {} samples".format(total))
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            # self._logger.debug('total_compute_time = {}'.format(total_compute_time))
            # self._logger.debug('iters_after_start = {}'.format(iters_after_start))
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    lvl=logging.INFO,
                    msg="Loss on Validation done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1,
                        total,
                        seconds_per_img,
                        str(eta)),
                    n=100,
                    name=__name__)

        mean_loss = np.mean(losses)

        self._logger.info("Validation loss : {}".format(mean_loss))
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return mean_loss


    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced


    def after_step(self):
        """
        TODO
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
