import argparse
import random
import pandas as pd
import numpy as np

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


def print_model_performance(model_performance):
    print('kfold,accuracy')
    accuracies = []
    for fold, accuracy in model_performance:
        print(f'{fold},{accuracy}')
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    print(f'mean,{accuracies.mean()}')
    print(f'standard deviation,{accuracies.std()}')


def main():
    random.seed(42)
    dataset = get_dataset()
    train = Train(
        dataset,
        get_target_column_from_argparser(dataset),
        get_ntree_from_argparser()
    )
    model_performance = train.execute()
    print_model_performance(model_performance)


if __name__ == '__main__':
    main()
