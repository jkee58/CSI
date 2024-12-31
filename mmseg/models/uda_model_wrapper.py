from typing import Any, Dict, Union
from overrides import overrides

import torch

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from mmengine.model import MMDistributedDataParallel
import torch.distributed as dist


@MODEL_WRAPPERS.register_module()
class UDAModelWrapper(MMDistributedDataParallel):

    def __init__(self, static_graph=False, **kwargs):
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        super().__init__(device_ids=device_id,
                         find_unused_parameters=True,
                         static_graph=static_graph,
                         **kwargs)

    @overrides
    def train_step(self, data,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        # with optim_wrapper.optim_context(self):
        # data = self.module.data_preprocessor(data, training=True)

        src_data, trg_data = data['src_data'], data['trg_data']
        data['trg_data'] = self.module.data_preprocessor(trg_data,
                                                         training=True)
        data['src_data'] = self.module.data_preprocessor(src_data,
                                                         training=True)

        optim_wrapper.zero_grad()
        log_vars = self._run_forward(data, mode='loss')  # type: ignore
        optim_wrapper.step()

        #     losses = self._run_forward(data, mode='loss')
        # parsed_loss, log_vars = self.module.parse_losses(losses)
        # optim_wrapper.update_params(parsed_loss)

        # remove the unnecessary 'loss'
        log_vars.pop('loss', None)

        return log_vars
