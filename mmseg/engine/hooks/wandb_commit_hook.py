from typing import Optional

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class WandbCommitHook(Hook):
    """Wandb Commit Hook. Used to sync step and iteration of wandb. Must be set to commit=False in WandbVisBackend.

    Args:
        interval (int): The interval of commit. Defaults to 1.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval

    @master_only
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            wandb = runner.visualizer.get_backend('WandbVisBackend').experiment
            wandb.log({'Iteration': runner.iter}, commit=True)
