#!/usr/bin/env python
import os
import cv2
import tqdm
import albumentations
import math
import random
import json
import copy
from .dataset_utility import *
from .base_dataset import BaseDataset


__all__ = ["HumanTrafficSignDataset"]


class HumanTrafficSignDataset(BaseDataset):
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

    def oversample_data(
        self,
        transformations: list,
        output_path: str,
        seed_value=2020,
        multiplication_factor=1.9,
        min_size=700,
    ):
        random.seed(seed_value)

        assert len(transformations) > 0

        image_num_map = {}
        for cls_name in self._classes:
            image_num_map[cls_name] = len(self._image_data_path_map[cls_name])
            print("class: {}, num images: {}".format(cls_name, len(self._image_data_path_map[cls_name])))

        data_distribution = self.get_data_distribution()
        max_num = max(data_distribution.values())
        max_num = int(multiplication_factor * max_num)
        oversample_num_map = dict(
            (cls_name, math.ceil((max_num - num_samples) / image_num_map[cls_name]))
            for (cls_name, num_samples) in data_distribution.items()
        )
        print(oversample_num_map)

        _bbox_params = albumentations.BboxParams(
            format="pascal_voc",
            min_area=0.001,
            min_visibility=0.001,
            label_fields=["category_ids"],
        )
        for cls_name in self._classes:
            if oversample_num_map[cls_name] == 0:
                continue
            os.system("mkdir -p {}".format(os.path.join(output_path, cls_name)))

            _transformations = copy.deepcopy(transformations)
            if cls_name == "Human":
                _transformations.extend(
                    [
                        albumentations.HorizontalFlip(p=0.7),
                        albumentations.Rotate(limit=10, p=0.5),
                    ]
                )

            for image_path, label_path in zip(
                tqdm.tqdm(self._image_data_path_map[cls_name]),
                self._label_data_path_map[cls_name],
            ):
                _cur_image = cv2.imread(image_path)
                image_height, image_width = _cur_image.shape[:2]

                _cur_bboxes, _cur_indices = self._load_one_ground_truth_file(label_path, box_normalized=False)

                # currently only apply this method to image with only one bbox
                if len(_cur_bboxes) > 1:
                    continue
                orig_bbox_width, orig_bbox_height = (
                    float(_cur_bboxes[0][2] - _cur_bboxes[0][0]),
                    float(_cur_bboxes[0][3] - _cur_bboxes[0][1]),
                )
                width_ratio = orig_bbox_width / image_width
                height_ratio = orig_bbox_height / image_height

                if max(width_ratio, height_ratio) > 0.8:
                    continue

                for i in range(oversample_num_map[cls_name]):
                    if image_height < min_size or image_width < min_size:
                        crop_ratio = random.uniform(0.7, 0.9)
                    else:
                        crop_ratio = random.uniform(0.3, 0.7)

                    crop_ratio = max(width_ratio, height_ratio, crop_ratio)

                    _crop_transform = albumentations.Compose(
                        [
                            albumentations.RandomCrop(
                                always_apply=True,
                                height=int(crop_ratio * image_height),
                                width=int(crop_ratio * image_width),
                            )
                        ],
                        bbox_params=_bbox_params,
                    )
                    _cur_transformations_obj = albumentations.Compose(_transformations, bbox_params=_bbox_params)

                    _transformed_bboxes = []
                    _transformed = None

                    aug_bbox_width, aug_bbox_height = 0.0, 0.0
                    try:
                        while (
                            len(_transformed_bboxes) == 0
                            or aug_bbox_width / orig_bbox_width < 0.9
                            or aug_bbox_height / orig_bbox_height < 0.9
                        ):
                            _transformed = _crop_transform(
                                image=_cur_image,
                                bboxes=_cur_bboxes,
                                category_ids=_cur_indices,
                            )
                            _transformed_bboxes = _transformed["bboxes"]
                            if len(_transformed_bboxes) > 0:
                                aug_bbox_width, aug_bbox_height = (
                                    float(_transformed_bboxes[0][2] - _transformed_bboxes[0][0]),
                                    float(_transformed_bboxes[0][3] - _transformed_bboxes[0][1]),
                                )

                        _transformed = _cur_transformations_obj(
                            image=_transformed["image"],
                            bboxes=_transformed["bboxes"],
                            category_ids=_transformed["category_ids"],
                        )
                        _base_name = image_path.split("/")[-1].split(".")[0] + "_sample_" + str(i)
                        _new_image_name = _base_name + ".jpg"
                        _new_json_name = _base_name + ".json"

                        _new_image_path = os.path.join(output_path, cls_name, _new_image_name)
                        _new_json_path = os.path.join(output_path, cls_name, _new_json_name)
                        cv2.imwrite(_new_image_path, _transformed["image"])

                        _new_img_height, _new_img_width = _transformed["image"].shape[:2]
                        _str_labels = [self._classes[label_idx] for label_idx in _transformed["category_ids"]]
                        _new_json_dict = create_label_me_json_dict(
                            _new_image_path,
                            _new_image_name,
                            _new_img_height,
                            _new_img_width,
                            _transformed["bboxes"],
                            _str_labels,
                        )
                        with open(_new_json_path, "w") as f:
                            json.dump(_new_json_dict, f)
                    except Exception as e:
                        print(e, " ", image_path)

    def _merge_all_data(self):
        for cls_name in self._classes:
            self._all_image_paths += self._image_data_path_map[cls_name]
            self._all_label_paths += self._label_data_path_map[cls_name]

        assert len(self._all_image_paths) == len(self._all_label_paths)

    def _load_one_ground_truth_file(self, json_path, box_normalized=True):
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
