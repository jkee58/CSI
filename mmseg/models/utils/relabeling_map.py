from dataclasses import dataclass, field
from collections.abc import MutableMapping
from collections import Counter
import pprint


@dataclass
class MappingElement:
    target_class: int
    source_candidates: Counter
    source_class: int = None
    meta_info: dict = field(
        default_factory=lambda: {
            'method': None,
            'threshold_for_detection': None,
            'threshold_for_classification': None,
            'origin_of_source_classes': None
        })


class RelabelingMap(MutableMapping):

    def __init__(self, relabeling_cfg, class_index_to_category):
        self._data = dict()

        for target_class, source_class, meta_info in relabeling_cfg:
            method, threshold_for_detection, threshold_for_classification = meta_info
            if source_class is None:
                origin_of_source_classes = 'auto'
            else:
                origin_of_source_classes = 'manual'

            meta_info = {
                'method': method,
                'threshold_for_detection': threshold_for_detection,
                'threshold_for_classification': threshold_for_classification,
                'origin_of_source_classes': origin_of_source_classes
            }

            self._data[target_class] = MappingElement(
                target_class=target_class,
                source_class=source_class,
                source_candidates=Counter(),
                meta_info=meta_info)

    def __getitem__(self, key) -> MappingElement:
        return self._data[key]

    def __setitem__(self, key: int, value: MappingElement):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> dict:
        return pprint.pformat(self._data)

    def clear(self):
        self._data.clear()

    def generate_detection_groups(self, class_index_to_category):
        return {
            class_index_to_category[class_index]
            for class_index in self._data.keys()
        }

    def get_elements_by_category(self, category_to_classes, category):
        results = dict()
        for index in (set(self._data.keys())
                      & set(category_to_classes[category])):
            results[index] = self._data[index]

        return results
