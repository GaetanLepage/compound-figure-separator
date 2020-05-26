"""
TODO
"""

import logging
from collections import defaultdict, OrderedDict

import torch

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm

class PanelSegAbstractEvaluator(DatasetEvaluator):
    """
    Class subclassing Detectron2's `DatasetEvaluator`.

    It is further subclassed for each specific task (panel splitting, label recognition,
    panel segmentation)
    """


    def __init__(self, dataset_name, task_name, evaluation_function):
        """
        init function.

        # TODO change default value for tasks
        # TODO manage output results in csv or xml or other
        # TODO test if outputed xml gives same results (metrics) if using the java tool from ImageCLEF

        Args:
            dataset_name (str):             name of the dataset
            task_name (str):                name of the task
            evaluation_function (function): function taking a figure generator as input and
                                                returning a dict of metric results
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = dict()

        self._task_name = task_name
        self._evaluation_function = evaluation_function

        # The figure generator corresponding to the dataset
        # TODO Not possible to handle LIST of test data sets.
        self._figure_generator = meta.figure_generator


    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = dict()


    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        This method is abstract and has to be implemented.

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """

        pass


    def _augmented_figure_generator(self, predictions):
        """
        Iterate over a Figure generator, process raw predictions and yield back the augmented
        Figure objects.

        This method is abstract and has to be implemented.
        """
        pass


    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key:      The name of the task ('panel_splitting', 'label_recognition',
                                'panel_segmentation')
                * value:    A dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        # Gathering
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            # TODO maybe return None instead of nothing
            return
        predictions = defaultdict(dict)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid] = lines
        del all_predictions

        # TODO remove
        for figure in self._augmented_figure_generator(predictions):
            pass

        # metrics_dict = self._evaluation_function(
            # figure_generator=self._augmented_figure_generator(predictions))

        # Respect the expected result for a DatasetEvaluator
        return OrderedDict({self._task_name: metrics_dict})
