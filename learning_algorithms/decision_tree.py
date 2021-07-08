import random
import string

def gini(frequencies):
    return 1 - np.sum(frequencies**2)


def entropy(frequencies, epsilon=1e-10):
    return np.sum(-frequencies*np.log2(frequencies + epsilon))


def misclass(frequencies):
    return 1 - np.max(frequencies)

def to_frequency(counts):
    return counts / np.sum(counts)


def compute_impurity_reduction(class_indicators, y, impurity_function):
    """
    args:
        class_indicators: (c, m). class indicator for this attribute division. class_indicators[i][j] indicates if example j is branch i
        y: (m, )
        impurity_function: gini, entropy, misclass
    return:
        impurity_reduction: impurity diff of this attribute division
    """
    frequency = to_frequency(np.array([np.sum(class_indicators, axis=1)]))
    frequencies = [to_frequency(np.unique(y[c], return_counts=True)[1]) for c in class_indicators]
    impurity = np.array([impurity_function(f) for f in frequencies])
    impurity_reduction = - (frequency @ impurity)
    return impurity_reduction

def get_matchers_by_equality(data):
    """
    args:
        data: (m, 1). data for a specific attribute
    returns:
        matchers: a dict whose key is name of branch, value is a function detect if the data match the branch by equality
    """
    classes = np.unique(data)
    return {"== " + str(c): lambda x, c=c: x == c for c in classes}


class Tree:
    def __init__(self, attr, label):
        self.attr = attr
        self.label = label
        self.branches = {}
        self.graph_id = ''.join(random.choices(string.ascii_uppercase, k=4))

    def add(self, branch, node):
        self.branches[branch] = node

    def digraph(self, level=0):
        out = ""
        out +=  "digraph G {\n rankdir=LR;\n" if level == 0 else ""
        out += = self.graph_id + "[label=\"(attr: %s, label: %s)\"];\n"%(str(self.attr), str(self.label))
        for b in self.branches:
            edge_label = "\"X[" + str(self.attr) + "] " + b + "\""
            _, child = self.branches[b]
            out += self.graph_id + " -> " + child.graph_id + "[label=" + edge_label + "];\n"
            out += child.digraph(level + 1)
        out += "}" if level == 0 else ""
        return head + out + tail


def ID3_train(X, y, attributes, impurity_function, get_matchers=get_matchers_by_equality):
    """
    args:
        X: (m, n)
        y: (m, 1)
        attributes: a list of attributes
        impurity_function: gini, entropy, misclass
        get_matchers: (data) -> [functions]. A function of computing matchers of branches for givin data.
    return:
        t: decision tree
    """

    classes, counts = np.unique(y, return_counts=True)
    label = np.squeeze(classes[np.argmax(counts)])

    # Stop criterion: 1. Nodes is clean; 2: Exhausted attributes
    if len(classes) == 1 or len(attributes) == 0:
        return Tree(None, label)

    # select the attribute with max impurity_reduction
    attr_branches = [(attr, \
                      matchers := get_matchers(X[:, attr]), \
                      np.array([matchers[k](X[:, attr]) for k in sorted(matchers.keys())])) \
                     for attr in attributes]
    attr, matchers, class_indicator = max(attr_branches,
                                          key=lambda x: compute_impurity_reduction(x[2], y, impurity_function))

    t = Tree(attr, label)
    new_attributes = [a for a in attributes if a != attr]
    for i, matcher_name in enumerate(sorted(matchers.keys())):
        indices = class_indicator[i]
        sub_tree = ID3(X[indices], y[indices], new_attributes, impurity_function)
        t.add(matcher_name, (matchers[matcher_name], sub_tree))
    return t


def ID3_predict_one(x, tree):
    for b in tree.branches:
        matcher, node = tree.branches[b]
        if matcher(x[tree.attr]):
            return ID3_predict_one(x, node)
    return tree.label


def ID3_predict(X, tree):
    return np.array([ID3_predict_one(x, tree) for x in X])
