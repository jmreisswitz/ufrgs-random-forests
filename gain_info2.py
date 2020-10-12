def divideset(data, column, value):
    split_function = None  # We splip the set for a value
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] == value
    else:
        split_function = lambda row: row[column] == value
    set1 = [row for row in data if split_function(row)]
    set2 = [row for row in data if not split_function(row)]
    return (set1, set2)

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


def buildtree(data, columns_already_used: set, scoref=entropy):
    #     if len(data) == 0:
    #         return decisionnode() #if the dataset is empty
    current_score = scoref(data)  # the entropy of the class

    # In this values we will chose the best.
    gain = 0.0
    best_gain = 0.0
    best_column = None

    column_count = len(data[0][:-1])  # all the columns less the lastone, the class column
    for col in range(0, column_count):
        if col in columns_already_used:
            continue
        column_values = {}
        gain = 0.0
        for row in data:
            column_values[row[col]] = 1  # count the values of each column
        for value in column_values.keys():  # the values will be the keys
            (set1, set2) = divideset(data, col, value)  # split the column for the value to test the entropy
            p = float(len(set1)) / len(data)  # p is the value respect the rest of the column.
            gain = gain + p * scoref(set1)  # acummulate of the entropy for each value in the column
        gain = current_score - gain  # the total gain of the column
        if gain >= best_gain and len(set1) != 0 and len(set2) != 0:  # we chose the biggest gain.
            best_gain = gain
            best_column = (col)
    return best_gain, best_column


# # this function receive the dataset and return the best column index and the Gain of the column
# best_gain, best_column = buildtree(my_data)
# print(best_gain)
# print(best_column)
# # print(best_sets[0])



