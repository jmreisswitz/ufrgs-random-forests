from abc import ABC, abstractmethod


class TreeNode(ABC):
    def __init__(self, column, condition):
        self.condition = condition
        self.children = []
        self.column = column

    @abstractmethod
    def predict_value(self, features):
        pass

    @staticmethod
    def get_tabs(depth):
        return '\t' * depth

    def print_node(self, columns_names, depth):
        print(f"{self.get_tabs(depth)}{self.condition} -> {self.__class__.__name__} {columns_names[self.column]}:")
        for child in self.children:
            child.print_node(columns_names, depth + 1)

    def __repr__(self):
        return_string = f'{self.condition} -> {self.__class__.__name__} {self.column}:'
        for child in self.children:
            return_string += f'\t{child}\n'
        return return_string[:-1]


class LeafNode(TreeNode):
    def __init__(self, condition, predicted_class):
        super().__init__(None, condition)
        self.predicted_class = predicted_class

    def print_node(self, columns_names, depth):
        print(f"{self.get_tabs(depth)}{self.condition} -> {self.__class__.__name__}: {self.predicted_class}")

    def predict_value(self, features):
        return self.predicted_class


class CategoricalNode(TreeNode):
    def __init__(self, condition, column, children, children_labels):
        super().__init__(column, condition)
        self.children_labels = children_labels
        self.children = children

    def predict_value(self, features):
        label_index = self.children_labels.index(features[self.column])
        return self.children[label_index].predict_value(features)


class NumericalNode(TreeNode):
    def __init__(self, column, condition, cutting_point, left_child: TreeNode, right_child: TreeNode):
        # condition here is only used for printing purposes
        super().__init__(column, condition)
        self.cutting_point = cutting_point
        self.children = [left_child, right_child]

    def predict_value(self, features):
        if self.cutting_point > features[self.column]:
            return self.children[0].predict_value(features)
        return self.children[1].predict_value(features)