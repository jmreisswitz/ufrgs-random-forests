import argparse
import random
import pandas as pd

from kfold import Kfold
from random_forest.random_forest import RandomForest
from train import Train

parser = argparse.ArgumentParser(description='Random Forests')
parser.add_argument('--ntree', type=int, default=20, help='Number of random forests on ensemble. Default = 20')
parser.add_argument('--dataset', type=str, default='dataset.csv', help='Dataset. Default = dataset.csv')
parser.add_argument('--target_column', type=str, default=None, help='Column to be predicted. Default = last column of dataset')
args = parser.parse_args()


def get_ntree_from_argparser():
    return args.ntree


def get_dataset_path_from_argparser():
    return f'datasets/{args.dataset}'


def get_target_column_from_argparser(dataset):
    if args.target_column is not None:
        return args.target_column
    return dataset.columns[-1]


def get_dataset():
    dataset_path = get_dataset_path_from_argparser()
    return pd.read_csv(dataset_path, sep='\t')


def main():
    random.seed(42)
    dataset = get_dataset()
    train = Train(
        RandomForest(get_ntree_from_argparser()),
        dataset,
        get_target_column_from_argparser(dataset)
    )
    model_performance = train.execute()
    print(f'Model got accuracy score = {model_performance}')


if __name__ == '__main__':
    main()
