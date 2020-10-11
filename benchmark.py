import pandas as pd
import numpy as np

from random_forest import RandomTree


def separate_dataset(dataset):
    benchmark_column = 'Joga'
    labels = np.array(dataset[benchmark_column])
    features = np.array(dataset.drop(benchmark_column, axis=1))
    return features, labels


def main():
    dataset = pd.read_csv('benchmark.csv', sep=';')
    features, labels = separate_dataset(dataset)
    print(features)
    print(labels)
    tree = RandomTree()
    tree.fit(features, labels)
    print(tree)


if __name__ == '__main__':
    main()
