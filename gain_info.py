from math import log


def log2(x):
    return log(x)/log(2)


class GainInfoService:
    def __init__(self, data, already_used_columns: set):
        self.already_used_columns = already_used_columns
        self.data = data

    @staticmethod
    def split_function(row, column, value):
        return row[column] == value

    def divide_set(self, column, value):
        set1 = [row for row in self.data if self.split_function(row, column, value)]
        set2 = [row for row in self.data if not self.split_function(row, column, value)]
        return set1, set2

    def unique_counts(self):
        results = {}
        for row in self.data:
            r = row[-1]  # we will count the values for the column.
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    def entropy(self, data):
        results = self.unique_counts()  # We count the classes for a set.
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(data)
            ent = ent - p * log2(p)  # We calculate the Entropy.
        return ent

    def build_tree(self):
        current_score = self.entropy(self.data)  # the entropy of the class

        # In this values we will chose the best.
        best_gain = 0.0
        best_column = None

        column_count = len(self.data[0][:-1])  # all the columns less the last one, the class column
        for col in range(0, column_count):
            if col in self.already_used_columns:
                continue
            column_values = {}
            gain = 0.0
            for row in self.data:
                column_values[row[col]] = 1  # count the values of each column
            for value in column_values.keys():  # the values will be the keys
                set1, set2 = self.divide_set(col, value)  # split the column for the value to test the entropy
                p = float(len(set1)) / len(self.data)  # p is the value respect the rest of the column.
                gain = gain + p * self.entropy(set1)  # acummulate of the entropy for each value in the column
            gain = current_score - gain  # the total gain of the column
            if gain >= best_gain and len(set1) != 0 and len(set2) != 0:  # we chose the biggest gain.
                best_gain = gain
                best_column = col
        return best_gain, best_column

    # # this function receive the dataset and return the best column index and the Gain of the column
    # best_gain, best_column = buildtree(my_data)
    # print(best_gain)
    # print(best_column)
    # # print(best_sets[0])
