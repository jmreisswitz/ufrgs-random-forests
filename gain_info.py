"""
Created on Mon Oct 12 01:15:39 2020

@author: eduar
"""


def divide_set(data, column, value):
    if isinstance(value, int) or isinstance(value, float):
        split_function= lambda row: row[column] <= value  # se o valor é int ou float.
    else:
        split_function=lambda row: row[column] == value  # se for string
    set1 = [row for row in data if split_function(row)]
    set2 = [row for row in data if not split_function(row)]
    return set1, set2


def uniquecounts(data):
    results = {}
    for row in data:
        r = row[-1]  # we will count the values for the column.
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def entropy(data):
    from math import log
    log2 = lambda x: log(x) / log(2)  # Base change for LOG2.
    results = uniquecounts(data)  # We count the classes for a set.
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(data)
        ent = ent - p * log2(p)  # We calculate the Entrophy.
    return ent


def buildtree(data, already_used_columns):
    current_score = entropy(data)  # the entropy of the class

    # In this values we will chose the best.
    best_gain = 0.0
    best_column = None
    best_value = None

    column_count = len(data[0][:-1])  # all the columns less the last one, the class column
    for col in range(0, column_count):
        if col in already_used_columns:
            continue
        column_values = {}
        gain = 0.0
        mean = 0.0
        if isinstance(data[0][col], int) or isinstance(data[0][col], float):
            for row in data:
                mean = mean + row[col]  # average
            mean = mean / len(data)
            set1, set2 = divide_set(data, col, mean)  # split the column for the value to test the entropy
            p1 = float(len(set1))/len(data)  # p is the value respect the rest of the column.
            p2 = float(len(set2))/len(data)  # p is the value respect the rest of the column.
            gain = current_score - ((p1 * entropy(set1)) + (p2 * entropy(set2)))  # acummulate of the entropy for each value in the column
            if gain >= best_gain and len(set1) != 0 and len(set2) != 0:  # we chose the biggest gain.
                best_gain = gain
                best_column = col
                best_value = mean
        else:
            for row in data:
                column_values[row[col]] = 1  # count the values of each column
            for value in column_values.keys():  # the values will be the keys
                set1, set2 = divide_set(data, col, value)  # split the column for the value to test the entropy
                p = float(len(set1)) / len(data)  # p is the value respect the rest of the column.
                gain = gain + (p * entropy(set1))  # acummulate of the entropy for each value in the column
            gain = current_score - gain  # the total gain of the column
            if gain >= best_gain and len(set1) != 0 and len(set2) != 0:  # we chose the biggest gain.
                best_gain = gain
                best_column = col
                best_value = value
    return best_gain, best_column, best_value
