import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype

from random_forest.random_tree import RandomTree


def separate_dataset(dataset):
    benchmark_column = 'Joga'
    labels = np.array(dataset[benchmark_column])
    features = np.array(dataset.drop(benchmark_column, axis=1))
    return features, labels


def get_columns_dict(dataset):
    columns_dict = dict()
    counter = 0
    for column in dataset.columns:
        columns_dict[counter] = column
        counter += 1
    return columns_dict


def predict_values(tree):
    # Type: [Tempo, Temperatura, umidade, ventoso]
    prediction = tree.predict(['Ensolarado', 'Quente', 'Alta', 'Falso'])
    print(f'Não = {prediction}')
    prediction = tree.predict(['Chuvoso', 'Quente', 'Alta', 'Falso'])
    print(f'Sim = {prediction}')
    prediction = tree.predict(['Nublado', 'Quente', 'Alta', 'Falso'])
    print(f'Sim = {prediction}')
    prediction = tree.predict(['Chuvoso', 'Quente', 'Alta', 'Verdadeiro'])
    print(f'Não = {prediction}')


def get_possible_features(dataset) -> dict:
    possibilities_dict = dict()
    counter = 0
    for column in dataset.columns:
        if not is_numeric_dtype(dataset[column]):
            possibilities_dict[counter] = set(dataset[column].unique())
        counter += 1
    return possibilities_dict


def main():
    dataset = pd.read_csv('datasets/benchmark.csv', sep=';')
    features, labels = separate_dataset(dataset)
    tree = RandomTree(get_possible_features(dataset))
    tree.fit(features, labels)
    tree.print_tree(get_columns_dict(dataset))
    predict_values(tree)


if __name__ == '__main__':
    main()
