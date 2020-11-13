#!/usr/bin/env python
import os
import abc
import cv2
import tqdm
from .dataset_utility import visualize_bboxes


__all__ = ["BaseDataset"]


class BaseDataset(abc.ABC):
    def __init__(self):
        self._all_image_paths = []
        self._all_label_paths = []
        self._classes = []
        self._colors = []

    def __len__(self):
        return len(self._all_image_paths)

    def visualize_one_image(self, idx):
        assert idx < self.__len__()
        image, all_bboxes, all_category_ids = self.__getitem__(idx, box_normalized=False)

        return visualize_bboxes(image, self._classes, self._colors, all_bboxes, all_category_ids)

    def __getitem__(self, idx, box_normalized=True):
        _cur_image = cv2.imread(self._all_image_paths[idx])
        _cur_bboxes, _cur_indices = self._load_one_ground_truth_file(
            self._all_label_paths[idx], box_normalized=box_normalized
        )
        return _cur_image, _cur_bboxes, _cur_indices

    def get_data_distribution(self):
        data_dist_map = dict((cur_label, 0) for cur_label in self._classes)

        for idx in tqdm.tqdm(range(self.__len__())):
            _, _cur_indices = self._load_one_ground_truth_file(self._all_label_paths[idx])
            for cur_idx in _cur_indices:
                data_dist_map[self._classes[cur_idx]] += 1

        return data_dist_map

    def create_yolo_train_data(self, output_path, sub_labels=[], ignore_ratio_thresh=1.0):
        if len(sub_labels) == 0:
            print("sub labels empty")
            return

        for label in sub_labels:
            if label not in self._classes:
                print("{} label does not exist in this dataset".format(label))
                return

        for image_path, label_path in zip(tqdm.tqdm(self._all_image_paths), self._all_label_paths):
            _base_name = image_path.split("/")[-1].split(".")[0]
            _new_label_name = _base_name + ".txt"
            _new_label_path = os.path.join(output_path, _new_label_name)

            _cur_bboxes, _cur_indices = self._load_one_ground_truth_file(label_path, box_normalized=True)

            write_content = ""
            for _cur_box, _cur_idx in zip(_cur_bboxes, _cur_indices):
                if self._classes[_cur_idx] not in sub_labels:
                    continue

                new_idx = sub_labels.index(self._classes[_cur_idx])

                xmin, ymin, xmax, ymax = self._mod_bbox(_cur_box)
                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                if width > ignore_ratio_thresh or height > ignore_ratio_thresh:
                    continue

                write_content += "{} {} {} {} {}\n".format(new_idx, xcenter, ycenter, width, height)

            if len(write_content) != 0:
                with open(_new_label_path, "w") as f:
                    f.write(write_content)
                    os.system("cp {} {}".format(image_path, output_path))

    @ property
    def classes(self):
        return self._classes

    @ abc.abstractmethod
    def _load_one_ground_truth_file(self, ground_truth_file_path, box_normalized=True):
        return NotImplemented

    def _mod_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        return [xmin, ymin, xmax, ymax]
