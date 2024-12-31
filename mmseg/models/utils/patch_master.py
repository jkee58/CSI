from typing import List, Deque, Dict
import random
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from collections import deque
from dataclasses import dataclass, astuple
from mmseg.models.utils.clip_guide import CLIPGuide
import torchvision


@dataclass
class Position:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class Patch:
    image: Tensor
    pseudo_label: Tensor
    weight: Tensor
    position: Position
    meta_info: dict
    segment_mask: Tensor = None
    relabeled_label: Tensor = None
    classification_confidence: float = .0
    detection_confidence: float = .0
    source_class: int = None
    target_class: int = None
    predicted_class: int = None
    full_image: Tensor = None
    full_pseudo_label: Tensor = None
    full_relabeled_label: Tensor = None
    full_src_image: Tensor = None

    def __post_init__(self):
        assert self.image.shape[1:] == self.pseudo_label.shape


class PatchBank:

    def __init__(self, patch_label, patch_capacity=100):
        self.patch_capacity: int = patch_capacity
        self.patch_label = patch_label
        self.vault: Deque[Patch] = deque(maxlen=patch_capacity)

    def withdraw_patch(self, method='random'):
        if method == 'random':
            return random.choice(self.vault)

    def deposit_patch(self, patch: Patch):
        self.vault.append(patch)

    def clear(self) -> None:
        self.vault.clear()


class Patcher:

    def __init__(self,
                 min_width=10,
                 min_height=10,
                 ignore_bottom=None,
                 ignore_top=None,
                 **kwargs):
        self.clip_guide = CLIPGuide(**kwargs)
        self.min_width = min_width
        self.min_height = min_height
        self.ignore_bottom = ignore_bottom
        self.ignore_top = ignore_top
        self.min_class_counts = 0.5
        self.threshold_for_class_ratio = 0.5
        self.threshold_for_validate_by_source_class = 0

    def get_roi(self, label, target_class, method):
        if len(label.shape) == 3 and label.shape[0] == 1:
            label = label.squeeze(0)
        assert len(label.shape) == 2

        y_min = torch.nonzero(label == target_class)[:, 0].min().item()
        y_max = torch.nonzero(label == target_class)[:, 0].max().item() + 1
        x_min = torch.nonzero(label == target_class)[:, 1].min().item()
        x_max = torch.nonzero(label == target_class)[:, 1].max().item() + 1
        height = y_max - y_min
        width = x_max - x_min

        return x_min, y_min, width, height

    def detect_classes(
        self,
        image: Tensor,
        mapping,
    ) -> List[tuple]:
        _, _, H, W = image.shape

        input_text, label_to_class = self.clip_guide.generate_text(
            mapping=mapping)
        results = self.clip_guide.predict(
            images=image,
            text=input_text,
            mapping=mapping,
            label_to_class=label_to_class,
            task='object_detection')

        # Post-process for boxes
        for result in results:
            result['boxes'] = torchvision.ops.clip_boxes_to_image(
                torch.round(result['boxes']), (H, W)).int()

        return results

    def extract_patches(self, image: Tensor, label: Tensor, weight: Tensor,
                        mapping) -> List[Patch]:

        results = self.detect_classes(image=image, mapping=mapping)

        assert len(results) == image.shape[0]

        batched_patches = []
        for batch, result in enumerate(results):
            patches = []
            for index, class_index in enumerate(result['labels'].tolist()):
                x_min, y_min, x_max, y_max = result['boxes'][index].tolist()
                height = y_max - y_min
                width = x_max - x_min

                meta_info = dict(
                    x_min=x_min,
                    y_min=y_min,
                    width=width,
                    height=height,
                    target_class=class_index,
                )
                patch = Patch(
                    image=F.crop(image[batch], y_min, x_min, height, width),
                    pseudo_label=F.crop(label[batch], y_min, x_min, height,
                                        width),
                    weight=F.crop(weight[batch], y_min, x_min, height, width),
                    position=Position(
                        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    meta_info=meta_info,
                    target_class=class_index,
                    source_class=mapping[class_index].source_class,
                    detection_confidence=result['scores'][index],
                    full_image=image[batch].clone(),
                    full_pseudo_label=label[batch].clone())
                patches.append(patch)
            batched_patches.append(patches)
        return batched_patches

    def validate_patch(self, patch: Patch, mapping) -> bool:
        height = patch.image.shape[0]
        y_min, width, height = patch.meta_info['y_min'], patch.meta_info[
            'width'], patch.meta_info['height']

        if width < self.min_width or height < self.min_height:
            return False

        if patch.predicted_class is not None:
            threshold_for_classification = mapping[patch.meta_info[
                'target_class']].meta_info['threshold_for_classification']
            if patch.predicted_class != patch.meta_info[
                    'target_class'] or patch.classification_confidence < threshold_for_classification:
                return False

        return True

    def classify_patches(self, batched_patches: List[List[Patch]], mapping):
        input_text, label_to_class = self.clip_guide.generate_text(
            mapping=mapping, mode='all_class')
        label_to_class = torch.tensor(label_to_class)
        images = [
            patch.image for patches in batched_patches for patch in patches
        ]

        if len(images) == 0:
            return batched_patches

        results = self.clip_guide.predict(
            images=images, text=input_text, task='classification')

        end = 0
        for patches in batched_patches:
            start = end
            end = start + len(patches)
            for index, patch in enumerate(patches):
                target_class = patch.target_class
                if mapping[target_class].meta_info[
                        'threshold_for_classification'] is not None:

                    category = self.clip_guide.class_index_to_category[
                        target_class]
                    classes_of_category = self.clip_guide.category_to_classes[
                        category]
                    category_indices = torch.nonzero(
                        torch.isin(label_to_class,
                                   torch.tensor(classes_of_category)))
                    class_probabilities = torch.softmax(
                        results[start:end][index][category_indices], dim=0)

                    patch.classification_confidence = round(
                        max(class_probabilities).item(), 4)
                    patch.predicted_class = label_to_class[category_indices][
                        torch.argmax(class_probabilities)].item()
        return batched_patches

    def get_classes_from_label(self, label, threshold=None):
        if threshold is None:
            threshold = self.threshold_for_class_ratio
        unique_valeus, counts = torch.unique(label, return_counts=True)
        return unique_valeus[counts / torch.numel(label) > threshold].tolist()

    def filter_patches(self,
                       batched_patches: List[List[Patch]],
                       validate_by: str,
                       mapping=None) -> List[List[Patch]]:

        def validate_by_pseudo_label(patch: Patch):
            count = torch.sum(
                torch.eq(patch.pseudo_label, patch.source_class)).item()
            if count > self.threshold_for_validate_by_source_class:
                return True
            else:
                return False

        def validate_by_classification_confidence(patch: Patch):
            if patch.predicted_class is None:
                return True

            threshold_for_classification = mapping[
                patch.target_class].meta_info['threshold_for_classification']

            if patch.predicted_class == patch.target_class and patch.classification_confidence > threshold_for_classification:
                return True
            else:
                return False

        def validate_by_patch_size(patch: Patch):
            width, height = patch.meta_info['width'], patch.meta_info['height']
            if width < self.min_width or height < self.min_height:
                return False
            else:
                return True

        if validate_by == 'pseudo_label':
            filter_function = validate_by_pseudo_label
        elif validate_by == 'classification_confidence':
            filter_function = validate_by_classification_confidence
        elif validate_by == 'patch_size':
            filter_function = validate_by_patch_size

        batched_patches = [
            list(filter(filter_function, patches))
            for patches in batched_patches
        ]

        return batched_patches

    def relabel_patch(self, patch: Patch):
        source_class, target_class = patch.source_class, patch.target_class
        if patch.segment_mask is None:
            patch.relabeled_label = torch.where(
                patch.pseudo_label == source_class, target_class,
                patch.pseudo_label)
        else:
            relabel_mask = torch.logical_and(
                patch.segment_mask, patch.pseudo_label == source_class)
            patch.relabeled_label = torch.where(relabel_mask, target_class,
                                                patch.pseudo_label)
        return patch

    def paste_patch(self, patch: Patch, pseudo_label: Tensor):
        x_min, y_min, x_max, y_max = astuple(patch.position)
        pseudo_label[y_min:y_max, x_min:x_max] = patch.relabeled_label

        return pseudo_label

    def get_source_candidate(self, patch: Patch):
        unique_values, counts = torch.unique(
            patch.pseudo_label, return_counts=True)
        return unique_values[counts / torch.numel(patch.pseudo_label) >
                             self.threshold_for_class_ratio].tolist()
