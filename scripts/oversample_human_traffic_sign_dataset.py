#!/usr/bin/env python
import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_DIR)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_path",
        type=str,
        required=True,
        help="path to the base directory of the dataset",
        dest="data_path",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        required=True,
        help="path to the augmented data",
        dest="output_path",
    )
    args = parser.parse_args()

    return args


def main():
    import albumentations
    from data_loader import HumanTrafficSignDataset

    args = get_args()

    transforms = [
        albumentations.MotionBlur(p=0.5),
        albumentations.Rotate(p=0.5, limit=(-5, 5)),
        albumentations.RandomBrightness(p=0.5, limit=(0.0, 0.2)),
        albumentations.IAASharpen(p=0.5, alpha=(0.05, 0.2)),
        albumentations.CLAHE(p=0.5),
        albumentations.RandomShadow(p=0.5),
        albumentations.FancyPCA(p=0.5),
        albumentations.RandomGamma(p=0.5),
        albumentations.RGBShift(p=0.5),
    ]

    dataset = HumanTrafficSignDataset(data_path=args.data_path, train=True)
    dataset.oversample_data(transforms, args.output_path)


if __name__ == "__main__":
    main()
