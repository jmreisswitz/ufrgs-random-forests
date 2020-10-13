
import pandas as pd
import numpy as np

from random_forest import RandomTree


def separate_dataset(dataset):
    benchmark_column = 'Outcome'
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
    # Type: [Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	Age	]
    prediction = tree.predict([0, 200, 0, 0, 0, 0, 0])
    print(f'1 = {prediction}')
    prediction = tree.predict([6, 100, 70, 0, 0, 0, 0])
    print(f'1 = {prediction}')
    prediction = tree.predict([5, 100, 70, 0, 0, 70, 0])
    print(f'0 = {prediction}')
    prediction = tree.predict([6, 100, 60, 0, 0, 28, 0])
    print(f'0 = {prediction}')
    prediction = tree.predict([6, 100, 60, 0, 45, 30, 29])
    print(f'1 = {prediction}')
    prediction = tree.predict([6, 100, 60, 0, 42, 30, 29])
    print(f'0 = {prediction}')
    prediction = tree.predict([6, 100, 60, 20, 42, 30, 31])
    print(f'0 = {prediction}')
    prediction = tree.predict([6, 100, 60, 18, 42, 30, 31])
    print(f'1 = {prediction}')


def main():
    dataset = pd.read_csv('diabetes.csv', sep=',')
    # pd.options.display.max_columns = dataset.shape[1]
    # print(dataset.describe())
    features, labels = separate_dataset(dataset)
    tree = RandomTree()
    tree.fit(features, labels)
    tree.print_tree(get_columns_dict(dataset))
    predict_values(tree)


if __name__ == '__main__':
    main()
