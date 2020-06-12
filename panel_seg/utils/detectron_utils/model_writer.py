"""
TODO
"""

from torch.utils.tensorboard import SummaryWriter

from detectron2.engine.hooks import HookBase



class ModelWriter(HookBase):
    """
    TODO
    """

    def __init__(self, model, input_example, log_dir):
        """
        TODO
        """
        self._model = model
        self._input_example = input_example
        self._log_dir = log_dir

    def before_train(self):

        SummaryWriter(log_dir=self._log_dir).add_graph(self._model,
                                                       self._input_example)
