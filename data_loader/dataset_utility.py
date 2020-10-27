#!/usr/bin/env python


__all__ = [
    "generate_color_chart",
    "human_sort",
    "get_all_files_with_format_from_path",
    "visualize_bboxes",
    "create_label_me_json_dict"
]


def generate_color_chart(num_classes, seed=3700):
    import numpy as np

    assert num_classes > 0
    np.random.seed(seed)

    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")
    colors = np.vstack([colors]).astype("uint8")
    colors = [tuple(color) for color in list(colors)]
    colors = [tuple(int(e) for e in color) for color in colors]

    return colors


def human_sort(s):
    """Sort list the way humans do"""
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def get_all_files_with_format_from_path(
    dir_path, suffix_format, concat_dir_path=True, use_human_sort=True
):
    import os

    all_files = [elem for elem in os.listdir(dir_path) if elem.endswith(suffix_format)]
    all_files.sort(key=human_sort)
    if concat_dir_path:
        all_files = [os.path.join(dir_path, cur_file) for cur_file in all_files]

    return all_files


def visualize_bboxes(image, classes, colors, all_bboxes, all_category_ids):
    import cv2

    for (bbox, label) in zip(all_bboxes, all_category_ids):
        x_min, y_min, x_max, y_max = [int(elem) for elem in bbox]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[label], 2)

        label_text = classes[label]
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_min + label_size[0][0], y_min + int(1.3 * label_size[0][1])),
            colors[label],
            -1,
        )
        cv2.putText(
            image,
            label_text,
            org=(x_min, y_min + label_size[0][1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )

    return image


def create_label_me_json_dict(img_path, img_name, img_height, img_width, bboxes, labels):
    import labelme
    import base64

    assert len(bboxes) == len(labels)

    json_dict = {}
    json_dict["imagePath"] = img_name
    json_dict["imageHeight"] = img_height
    json_dict["imageWidth"] = img_width
    json_dict["imageData"] = base64.b64encode(labelme.LabelFile.load_image_file(img_path)).decode('utf-8')
    json_dict["shapes"] = []
    for bbox, label in zip(bboxes, labels):
        cur_shape = {}
        cur_shape["label"] = label
        cur_shape["shape_type"] = "rectangle"
        xmin, ymin, xmax, ymax = bbox
        cur_shape["points"] = [[xmin, ymin], [xmax, ymax]]
        cur_shape["group_id"] = None
        cur_shape["flags"] = {}

        json_dict["shapes"].append(cur_shape)

    return json_dict
