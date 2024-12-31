from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS

@HOOKS.register_module()
class IterationHook(Hook):
    """Docstring for CustomHook.
    """

    def before_train_iter(self, runner, batch_idx: int, data_batch: Optional[Sequence[dict]] = None) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        # model.local_iter = runner.iter