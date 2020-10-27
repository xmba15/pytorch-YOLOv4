#!/usr/bin/env python
import os
import cv2
from torch.utils.data.dataset import Dataset
from .dataset_utility import *


__all__ = ["HumanTrafficSignDataset"]


class HumanTrafficSignDataset(Dataset):
    def __init__(self, data_path, train=True):
        super(HumanTrafficSignDataset, self).__init__()

        self._train = train
        self._classes = ["Human", "SlowStart", "SlowEnd", "Stop"]
        self._colors = generate_color_chart(len(self._classes), seed=2020)

        assert os.path.isdir(data_path)
        self._data_path = data_path
        if self._train:
            self._data_path = os.path.join(self._data_path, "TrainDB")
        else:
            self._data_path = os.path.join(self._data_path, "ValidationDB")

        self._image_data_path_map = {}
        self._label_data_path_map = {}
        self._labels_map = {}

        for cls_name in self._classes:
            self._image_data_path_map[cls_name] = get_all_files_with_format_from_path(
                os.path.join(self._data_path, cls_name), ".jpg"
            )
            self._label_data_path_map[cls_name] = get_all_files_with_format_from_path(
                os.path.join(self._data_path, cls_name), ".json"
            )
            assert len(self._image_data_path_map[cls_name]) == len(self._label_data_path_map[cls_name])

        self._merge_all_data()

    def visualize_one_image(self, idx):
        assert idx < self.__len__()
        image, all_bboxes, all_category_ids = self.__getitem__(
            idx, box_normalized=False
        )

        return visualize_bboxes(
            image, self._classes, self._colors, all_bboxes, all_category_ids
        )

    def __getitem__(self, idx, box_normalized=True):
        _cur_image = cv2.imread(self._all_image_paths[idx])
        _cur_bboxes, _cur_indices = self._load_one_json(
            self._all_label_paths[idx], box_normalized=box_normalized
        )
        return _cur_image, _cur_bboxes, _cur_indices

    def __len__(self):
        return len(self._all_image_paths)

    def _merge_all_data(self):
        self._all_image_paths = []
        self._all_label_paths = []

        for cls_name in self._classes:
            self._all_image_paths += self._image_data_path_map[cls_name]
            self._all_label_paths += self._label_data_path_map[cls_name]

        assert len(self._all_image_paths) == len(self._all_label_paths)

    def _load_one_json(self, json_path, box_normalized=True):
        import json

        json_dict = {}
        with open(json_path) as f:
            json_dict = json.load(f)
        assert len(json_dict) != 0

        bboxes = []
        label_indices = []

        height = json_dict["imageHeight"]
        width = json_dict["imageWidth"]
        assert height > 0 and width > 0

        for obj in json_dict["shapes"]:
            cls_name = obj["label"]
            assert cls_name in self._classes
            cls_idx = self._classes.index(cls_name)

            bbox = obj["points"]
            xmin, ymin = bbox[0]
            xmax, ymax = bbox[1]

            if box_normalized:
                xmin /= width
                ymin /= height
                xmax /= width
                ymax /= height

            bboxes.append([xmin, ymin, xmax, ymax])
            label_indices.append(cls_idx)

        return bboxes, label_indices
