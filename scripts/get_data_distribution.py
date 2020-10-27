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
    args = parser.parse_args()

    return args


def main():
    from data_loader import HumanTrafficSignDataset

    args = get_args()

    dataset = HumanTrafficSignDataset(data_path=args.data_path, train=True)
    print(dataset.get_data_distribution())


if __name__ == "__main__":
    main()
