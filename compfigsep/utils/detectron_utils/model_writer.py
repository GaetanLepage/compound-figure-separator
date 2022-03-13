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


###########################################################################
Custom hook to write the model graph (architecture) in TensorBoard storage.
"""

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from detectron2.engine.hooks import HookBase


class ModelWriter(HookBase):
    """
    Custom hook to write the model graph (architecture) in TensorBoard storage.
    """

    def __init__(self,
                 model: nn.Module,
                 input_example: dict,
                 log_dir: str):
        """
        Init function for the ModelWriter.

        Args:
            model (nn.Module):      The model to display in Tensorboard.
            input_example (dict):   An exemple of a model input to infer the input shape.
            log_dir (str):          The path to the folder where to store the output log.
        """
        self._model: nn.Module = model
        self._input_example: dict = input_example
        self._log_dir: str = log_dir

    def before_train(self) -> None:
        """
        Called at the beginning of the training process.
        Write the model shape to the summary writer.
        """
        SummaryWriter(log_dir=self._log_dir).add_graph(self._model,
                                                       self._input_example)
