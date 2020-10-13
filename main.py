import argparse
import random
import pandas as pd

from kfold import Kfold
from random_forest.random_forest import RandomForest
from train import Train

parser = argparse.ArgumentParser(description='Random Forests')
parser.add_argument('--ntree', type=int, default=20, help='Number of random forests on ensemble. Default = 20')
parser.add_argument('--dataset', type=str, default='dataset.tsv', help='Dataset. Default = dataset.csv')
parser.add_argument('--target_column', type=str, default='target', help='Column to be predicted. Default = target')
args = parser.parse_args()


def get_ntree_from_argparser():
    return args.ntree


def get_dataset_path_from_argparser():
    return args.dataset


def get_target_column_from_argparser():
    return args.target_columns


def get_dataset():
    dataset_path = get_dataset_path_from_argparser()
    return pd.read_csv(dataset_path, sep='\t')


def main():
    random.seed(42)
    train = Train(
        RandomForest(get_ntree_from_argparser()),
        get_dataset(),
        get_target_column_from_argparser()
    )
    train.execute()


if __name__ == '__main__':
    main()
