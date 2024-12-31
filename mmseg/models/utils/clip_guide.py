from typing import List
import torch
from torch import Tensor
from transformers import CLIPProcessor, CLIPModel
from mmseg.models.utils.dacs_transforms import denorm
from mmengine.logging import print_log
import torchvision

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import SamModel, SamProcessor


class CLIPGuide:

    def __init__(self,
                 mean,
                 std,
                 checkpoint_for_object_detection,
                 checkpoint_for_segment='sam'):
        # move model to device if possible
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

        # Load CLIP for classification
        self.checkpoint_for_classification = "openai/clip-vit-base-patch32"
        self.model_for_classification = CLIPModel.from_pretrained(
            self.checkpoint_for_classification).to(self.device)
        self.processor_for_classification = CLIPProcessor.from_pretrained(
            self.checkpoint_for_classification, do_rescale=False)

        # Load model for object detection
        if checkpoint_for_object_detection == 'owlvit':
            self.checkpoint_for_object_detection = 'google/owlvit-base-patch32'
            self.model_for_object_detection = OwlViTForObjectDetection.from_pretrained(
                self.checkpoint_for_object_detection).to(self.device)
            self.processor_for_object_detection = OwlViTProcessor.from_pretrained(
                self.checkpoint_for_object_detection, do_rescale=False)
        elif checkpoint_for_object_detection == 'owlv2':
            self.checkpoint_for_object_detection = 'google/owlv2-base-patch16-ensemble'
            self.model_for_object_detection = Owlv2ForObjectDetection.from_pretrained(
                self.checkpoint_for_object_detection).to(self.device)
            self.processor_for_object_detection = Owlv2Processor.from_pretrained(
                self.checkpoint_for_object_detection, do_rescale=False)
        else:
            raise NotImplementedError(
                'Supported object detection checkpoints: owlvit, owlv2')

        # Load model for segment
        if checkpoint_for_segment == 'sam':
            self.checkpoint_for_segment = 'facebook/sam-vit-huge'
            self.model_for_segment = SamModel.from_pretrained(
                self.checkpoint_for_segment).to(self.device)
            self.processor_for_segment = SamProcessor.from_pretrained(
                self.checkpoint_for_segment, do_rescale=False)
        else:
            raise NotImplementedError('Supported segment checkpoints: sam')

        # Generate map for class
        self.class_index_to_category = self.get_class_index_to_category()
        self.class_index_from_label_of_prompt = self.get_class_index_from_label_of_prompt(
            cityscapes_classes_with_concept)
        print_log(
            f"Generated mapping dictionary (class to label of prompt): {self.class_index_from_label_of_prompt}",
            logger="current")
        self.category_to_classes = self.get_category_to_classes(
            self.class_index_from_label_of_prompt)
        print_log(
            f"Generated mapping dictionary (category to classes): {self.category_to_classes}",
            logger="current")

    def get_class_index_from_label_of_prompt(self, classes_with_concept):
        class_index_from_label_of_prompt = dict()
        for class_index, concepts in enumerate(classes_with_concept):
            category = self.class_index_to_category[class_index]
            if class_index_from_label_of_prompt.get(category, None) is None:
                class_index_from_label_of_prompt[category] = [class_index
                                                              ] * len(concepts)
            else:
                class_index_from_label_of_prompt[category].extend(
                    [class_index] * len(concepts))

        return class_index_from_label_of_prompt

    def get_class_index_to_category(self, dataset='cityscapes'):
        if dataset == 'cityscapes':
            return {
                0: 'flat',
                1: 'flat',
                2: 'construction',
                3: 'construction',
                4: 'construction',
                5: 'object',
                6: 'object',
                7: 'object',
                8: 'nature',
                9: 'nature',
                10: 'sky',
                11: 'human',
                12: 'human',
                13: 'vehicle',
                14: 'vehicle',
                15: 'vehicle',
                16: 'vehicle',
                17: 'vehicle',
                18: 'vehicle'
            }

    def get_category_to_classes(self, class_index_from_label_of_prompt):
        category_to_classes = dict()
        for category, classes in class_index_from_label_of_prompt.items():
            category_to_classes[category] = list(dict.fromkeys(classes))

        return category_to_classes

    def predict(self,
                images,
                mapping: dict = None,
                label_to_class: list = None,
                task='classification',
                **kwargs):
        images = self.denormalize_image_for_model(images)
        with torch.no_grad():
            if task == 'classification':
                text = kwargs['text']
                inputs = self.processor_for_classification(
                    images=images,
                    return_tensors="pt",
                    text=text,
                    padding=True).to(self.device)
                outputs = self.model_for_classification(**inputs)
                results = self.model_for_classification(
                    **inputs).logits_per_image
            elif task == 'object_detection':
                B, _, H, W = images.shape
                min_threshold = min([
                    element.meta_info['threshold_for_detection']
                    for element in mapping.values()
                ])
                text = kwargs['text']
                inputs = self.processor_for_object_detection(
                    images=images, return_tensors="pt",
                    text=[text] * B).to(self.device)
                outputs = self.model_for_object_detection(**inputs)
                target_sizes = torch.tensor([[H, W]]).expand(B, -1)
                results = self.processor_for_object_detection.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=min_threshold)

                for result in results:
                    # Convert label with class index and remove data with threshold for each class
                    retain_indices = []
                    for index, label in enumerate(result['labels']):
                        class_index = label_to_class[label]
                        result['labels'][index] = class_index
                        if result['scores'][index] > mapping[
                                class_index].meta_info[
                                    'threshold_for_detection']:
                            retain_indices.append(index)

                    scores = result['scores'][retain_indices]
                    labels = result['labels'][retain_indices]
                    boxes = result['boxes'][retain_indices]

                    nms_indices = torchvision.ops.nms(
                        boxes, scores, iou_threshold=0.3)

                    scores = scores[nms_indices]
                    labels = labels[nms_indices]
                    boxes = boxes[nms_indices]

                    result['scores'], sort_indices = torch.sort(scores)
                    result['labels'] = labels[sort_indices]
                    result['boxes'] = boxes[sort_indices]

                    assert len(result['scores']) == len(
                        result['labels']) == len(result['boxes'])
            elif task == 'segment':
                input_boxes = kwargs['input_boxes']
                inputs = self.processor_for_segment(
                    images, input_boxes=input_boxes,
                    return_tensors="pt").to(self.device)
                outputs = self.model_for_segment(
                    **inputs, multimask_output=False)
                masks = self.processor_for_segment.image_processor.post_process_masks(
                    outputs.pred_masks, inputs["original_sizes"],
                    inputs["reshaped_input_sizes"])
                scores = outputs.iou_scores
                return masks
            else:
                raise NotImplementedError(
                    'Supported tasks: classification, object_detection, segment'
                )

            return results

    def generate_text(self, mapping, patch=None, mode='all_target'):
        concepts = []
        label_to_class = []

        if mode == 'all_class':
            for class_index, concept in enumerate(
                    cityscapes_classes_with_concept):
                concepts.extend(concept)
                label_to_class.extend([class_index] * len(concept))
        elif mode == 'all_target':
            for target_class in mapping.keys():
                concept = cityscapes_classes_with_concept[target_class]
                concepts.extend(concept)
                label_to_class.extend([target_class] * len(concept))
        elif mode == 'target_category':
            assert patch is not None
            category = self.class_index_to_category[patch.target_class]
            classes_of_target_category = self.category_to_classes[category]
            for class_index in classes_of_target_category:
                concept = cityscapes_classes_with_concept[class_index]
                concepts.extend(concept)
                label_to_class.extend([class_index] * len(concept))

        assert len(concepts) == len(label_to_class)
        return concepts, label_to_class

    # def generate_text(self, category=None, mode='all_of_target_category'):
    #     if mode == 'all_class':
    #         pass
    #     elif mode == 'all_of_target_category':
    #         assert category is not None
    #         classes_of_target_category = self.category_to_classes[category]
    #         concepts = []
    #         for class_index in classes_of_target_category:
    #             concepts.extend(cityscapes_classes_with_concept[class_index])

    #         return concepts

    # elif mode == 'only_target_class':
    #     target_class_text = cityscapes_classes_with_concept[target_class]
    #     return [f"{text}"
    #             for text in target_class_text], len(target_class_text)
    # elif mode == 'concat_target_with_other':
    #     target_class_text = cityscapes_classes_with_concept[target_class]
    #     other_class_text = cityscapes_classes_with_concept[other_class]
    #     concated_text = [f"{text}" for text in target_class_text
    #                      ] + [f"{text}" for text in other_class_text]

    #     return concated_text, len(target_class_text)
    # elif mode == 'concat_target_with_all':
    #     target_class_text = cityscapes_classes_with_concept[target_class]
    #     other_class_text = []
    #     for class_index, concept in enumerate(
    #             cityscapes_classes_with_concept):
    #         if class_index != target_class:
    #             other_class_text.extend(concept)
    #     concated_text = [f"{text}" for text in target_class_text
    #                      ] + [f"{text}" for text in other_class_text]

    #     return concated_text, len(target_class_text)
    # elif mode == 'concat_target_with_category':
    #     target_class_text = cityscapes_classes_with_concept[target_class]
    #     other_class_of_target_category = self.get_other_class(target_class)
    #     other_class_text = []
    #     for class_index, concept in enumerate(
    #             cityscapes_classes_with_concept):
    #         if class_index in other_class_of_target_category:
    #             other_class_text.extend(concept)
    #     concated_text = [f"{text}" for text in target_class_text
    #                      ] + [f"{text}" for text in other_class_text]

    #     return concated_text, len(target_class_text)
    # else:
    #     raise NotImplementedError(
    #         'Supported modes: all_class, only_target_class, concat_target_with_other, concat_target_with_all'
    #     )

    def denormalize_image_for_model(self, images):
        if isinstance(images, list):
            return [
                torch.clamp(denorm(image, self.mean, self.std), 0, 1)
                for image in images
            ]

        return torch.clamp(denorm(images, self.mean, self.std), 0, 1)

    # def validate_image(self, image, from_class, to_class):
    #     image_for_clip = self.denormalize_image_for_clip(image)
    #     text_for_clip, from_text_index = self.generate_text_for_clip(
    #         from_class, to_class)
    #     with torch.no_grad():
    #         class_probabilities = self.forward(image_for_clip, text_for_clip,
    #                                            from_text_index)
    #     return class_probabilities

    def generate_text_for_clip(self, from_class, to_class):
        to_text = cityscapes_classes_with_concept[to_class]

        if from_class is None:
            return [text for text in to_text], None
        else:
            from_text = cityscapes_classes_with_concept[from_class]
            text_for_clip = [text for text in from_text
                             ] + [text for text in to_text]
            return text_for_clip, len(from_text)

    def forward(self, image_for_clip: Tensor, text_for_clip, from_text_index):
        inputs = self.processor(
            images=image_for_clip,
            return_tensors="pt",
            text=text_for_clip,
            padding=True).to(self.device)

        logits_per_image = self.model(**inputs).logits_per_image
        label_probabilities = logits_per_image.softmax(dim=1).squeeze(0)
        return torch.tensor([
            torch.sum(label_probabilities[:from_text_index]),
            torch.sum(label_probabilities[from_text_index:])
        ])

    def predict_by_owl(self, image_for_owl, text_for_owl):
        with torch.no_grad():
            inputs = self.processor_for_detect(
                images=image_for_owl, return_tensors="pt",
                text=text_for_owl).to(self.device)
            outputs = self.model_for_detect(**inputs)
            target_sizes = torch.tensor([image_for_owl.size[::-1]])
            results = self.processor_for_detect.post_process_object_detection(
                outputs, threshold=0.01, target_sizes=target_sizes)[0]

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        return scores, labels, boxes

    def get_other_class(self, target_class):
        class_index_to_category = {
            0: 'flat',
            1: 'flat',
            2: 'construction',
            3: 'construction',
            4: 'construction',
            5: 'object',
            6: 'object',
            7: 'object',
            8: 'nature',
            9: 'nature',
            10: 'sky',
            11: 'human',
            12: 'human',
            13: 'vehicle',
            14: 'vehicle',
            15: 'vehicle',
            16: 'vehicle',
            17: 'vehicle',
            18: 'vehicle'
        }

        category_to_class_index = {
            'flat': [0, 1],
            'construction': [2, 3, 4],
            'object': [5, 6, 7],
            'nature': [8, 9],
            'sky': [10],
            'human': [11, 12],
            'vehicle': [13, 14, 15, 16, 17, 18]
        }

        target_category_class = category_to_class_index[
            class_index_to_category[target_class]]
        target_category_class.remove(target_class)

        return target_category_class

    # def get_category_to_classes(self, category):
    #     category_to_class_index = {
    #         'flat': [0, 1],
    #         'construction': [2, 3, 4],
    #         'object': [5, 6, 7],
    #         'nature': [8, 9],
    #         'sky': [10],
    #         'human': [11, 12],
    #         'vehicle': [13, 14, 15, 16, 17, 18]
    #     }

    #     return category_to_class_index[category]

    def get_category_by_class_index(self, class_index):
        class_index_to_category = {
            0: 'flat',
            1: 'flat',
            2: 'construction',
            3: 'construction',
            4: 'construction',
            5: 'object',
            6: 'object',
            7: 'object',
            8: 'nature',
            9: 'nature',
            10: 'sky',
            11: 'human',
            12: 'human',
            13: 'vehicle',
            14: 'vehicle',
            15: 'vehicle',
            16: 'vehicle',
            17: 'vehicle',
            18: 'vehicle'
        }

        return class_index_to_category[class_index]

    def get_indices_from_category(self, target_class_index, category):
        start_index_of_class = 0
        category = self.get_category_by_class_index(target_class_index)
        classes_of_category = self.get_category_to_classes(category)
        for class_index in classes_of_category:
            if class_index == target_class_index:
                break

            start_index_of_class += len(
                cityscapes_classes_with_concept[class_index])

        return list(
            range(
                start_index_of_class, start_index_of_class +
                len(cityscapes_classes_with_concept[target_class_index])))

    def convert_classes_to_labels_of_category(self, target_classes: List[int],
                                              category: str) -> List[int]:
        labels_of_category = []
        start_index_of_class = 0
        for class_index in self.category_to_classes[category]:
            if class_index in target_classes:
                labels_of_category.extend(
                    list(
                        range(
                            start_index_of_class, start_index_of_class +
                            len(cityscapes_classes_with_concept[class_index])))
                )
            start_index_of_class += len(
                cityscapes_classes_with_concept[class_index])

        return labels_of_category


# TODO: Refact
cityscapes_classes_with_concept = [
    ['road', 'street', 'parking space', 'tram track'],
    ['sidewalk'],
    [
        'building', 'skyscaper', 'house', 'bus stop building', 'garage',
        'car port', 'scaffolding'
    ],
    ['standing wall, which is not part of a building', 'retaining wall'],
    ['fence', 'hole in fence'],
    ['pole', 'sign pole', 'traffic light pole'],
    ['traffic light'],
    ['traffic sign', 'parking sign', 'direction sign'],
    ['vegetation', 'tree', 'hedge'],
    ['terrain', 'grass', 'soil', 'sand', 'roadside grass'],
    ['sky'],
    [
        'person', 'pedestrian', 'walking person', 'standing person',
        'person sitting on the ground', 'person sitting on a bench',
        'person sitting on a chair'
    ],
    ['rider', 'cyclist', 'motorcyclist', 'person on bicycle', 'person on motorbike', 'person on bike'],
    ['car', 'jeep', 'SUV', 'van'],
    ['truck', 'box truck', 'pickup truck', 'truck trailer'],
    ['bus'],
    ['train', 'tram'],
    ['motorcycle', 'moped', 'scooter'],
    ['bicycle'],
]
