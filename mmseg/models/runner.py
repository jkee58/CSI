from mmengine.registry import MODELS
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from typing import Dict, Union
import torch.nn as nn
import torch


@RUNNERS.register_module()
class CustomRunner(Runner):

    def __init__(self, **args):
        super().__init__(**args)

    def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
        """Build model.

        If ``model`` is a dict, it will be used to build a nn.Module object.
        Else, if ``model`` is a nn.Module object it will be returned directly.

        An example of ``model``::

            model = dict(type='ResNet')

        Args:
            model (nn.Module or dict): A ``nn.Module`` object or a dict to
                build nn.Module object. If ``model`` is a nn.Module object,
                just returns itself.

        Note:
            The returned model must implement ``train_step``, ``test_step``
            if ``runner.train`` or ``runner.test`` will be called. If
            ``runner.val`` will be called or ``val_cfg`` is configured,
            model must implement `val_step`.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if 'uda' in self.cfg:
            self.cfg.uda['model'] = self.cfg.model
            self.cfg.uda['max_iters'] = self.max_iters
            return MODELS.build(self.cfg.uda,
                                default_args=dict(train_cfg=self.cfg.train_cfg,
                                                  test_cfg=self.cfg.test_cfg))
        elif isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            model = MODELS.build(model)
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')
