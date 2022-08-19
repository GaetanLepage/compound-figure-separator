import torch

from yolox.exp import Exp


class PanelSplittingExp(Exp):
    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10

    def get_data_loader(
        self,
        batch_size: int,
        is_distributed: bool
    ) -> dict[str, torch.utils.data.DataLoader]:

        data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(

        )

    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    def get_evaluator(self):
        pass

    def eval(self, model, evaluator, weights):
        pass
