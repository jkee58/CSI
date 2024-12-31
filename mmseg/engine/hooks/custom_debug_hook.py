import io

from typing import Optional, Sequence

import torch
from torch import Tensor

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.dist import master_only

from mmseg.registry import HOOKS

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


@HOOKS.register_module()
class CustomDebugHook(Hook):
    """Docstring for CustomHook.
    """

    def __init__(self, interval: int = 1000):
        self.interval = interval
        self.model = None
        self.visualizer = None

    def before_run(self, runner) -> None:
        if is_model_wrapper(runner.model):
            self.model = runner.model.module
        else:
            self.model = runner.model
        self.model.debug_img = False
        self.visualizer = self.model.seg_visualizer

    @master_only
    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            self.model.debug_img = True
        self.model.update_debug_state(self.model.debug_img)

    @master_only
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:

        if not hasattr(self.model, 'custom_debug'):
            return

        if self.model.debug_img == True:
            self.model.debug_img = False

        custom_debug = self.model.custom_debug

        for figure_name, content in custom_debug.items():
            figure_info = content['figure_info']
            rows, cols = figure_info['shape']
            subplot_size = figure_info.get('subplot_size', 3)
            figure_size = figure_info.get(
                'figure_size', (subplot_size * cols, subplot_size * rows))
            figure_kwargs = figure_info.get('kwargs', {})

            plots_info = content['plots_info']
            figure, axes = plt.subplots(
                rows, cols, figsize=figure_size, **figure_kwargs)
            for plot_info in plots_info:
                i = plot_info.pop('axis', None)
                if rows == 1:
                    self.subplot_image(axis=axes[i[1]], **plot_info)
                else:
                    self.subplot_image(axis=axes[i], **plot_info)

            self.remove_axes(axes)
            if figure_kwargs == {}:
                figure.tight_layout()
            figure_image = self.convert_figure_to_numpy(figure)
            self.visualizer.add_image(
                name=figure_name, image=figure_image, step=runner.iter)

    def convert_figure_to_numpy(self, figure):
        with io.BytesIO() as figure_buffer:
            plt.savefig(figure_buffer, format='raw')
            plt.close()
            image = np.reshape(
                np.frombuffer(figure_buffer.getvalue(), dtype=np.uint8),
                newshape=(int(figure.bbox.bounds[3]),
                          int(figure.bbox.bounds[2]), -1))
        return image

    def subplot_image(self, axis, image: Tensor, title=None, **kwargs):
        if title is not None:
            axis.set_title(title)

        image = self.convert_image_tensor_to_numpy(image)

        heatmap = kwargs.pop('heatmap', None)
        scatter = kwargs.pop('scatter', None)

        cmap = kwargs.pop('cmap', None)
        if cmap == 'cityscapes':
            image = self.colorize_mask(image, CITYSCAPES_PALLETE)
            cmap = None

        axis.imshow(image, cmap=cmap, **kwargs)

        if isinstance(heatmap, dict):
            heatmap_image = self.convert_image_tensor_to_numpy(
                heatmap['image'])
            alpha = heatmap.get('alpha', .5)
            symmetric = heatmap.get('symmetric')
            cmap = heatmap.get('cmap', 'bwr')

            if symmetric:
                vmax = np.abs(heatmap_image).max()
                heatmap_kwargs = dict(vmax=vmax, vmin=-vmax)
            heatmap_kwargs = {}

            axis.imshow(
                heatmap_image, alpha=alpha, cmap=cmap, **heatmap_kwargs)

        if isinstance(scatter, dict):
            axis.scatter(**scatter)

    def convert_image_tensor_to_numpy(self, image: Tensor):
        if torch.is_tensor(image):
            with torch.no_grad():
                if image.shape[0] == 1:
                    image = image.squeeze(0)
                elif image.shape[0] == 3:
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()

        elif not isinstance(image, np.ndarray):
            raise NotImplementedError(type(image))

        return image

    def remove_axes(self, axes):
        for ax in axes.flat:
            ax.axis('off')

    def colorize_mask(self, mask, palette):
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask


CITYSCAPES_PALLETE = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
    153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
    180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
    0, 0, 230, 119, 11, 32, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128,
    128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128,
    192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
    64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64,
    0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64,
    128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64,
    0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64,
    64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192,
    192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128,
    160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0,
    224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64,
    0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192,
    128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64,
    128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
    128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
    192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0,
    192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64,
    160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96,
    64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0,
    0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32,
    0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192,
    160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128,
    96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0,
    192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32,
    64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
    160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
    64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128,
    96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
    128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
    192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
    160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160,
    128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160,
    224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0,
    224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
    128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32,
    32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32,
    64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192,
    224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96,
    192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64,
    96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 0, 0, 0
]
