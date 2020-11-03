import numpy as np

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)

        self.tree = Tree()

        if features.size == 0:
            self.tree.value = np.argmax(np.bincount(targets.astype(int)))
            return
        if np.unique(targets).size == 1:
            self.tree.value = np.unique(targets)[0]
            return

        gain_arr = np.zeros(features.shape[1])
        for i in range(features.shape[1]):
            gain_arr[i] = information_gain(features, i, targets)

        best_attr = np.argmax(gain_arr)

        self.tree.attribute_name = self.attribute_names[best_attr]
        self.tree.attribute_index = best_attr
        attr_col = features[:, best_attr]

        features = np.delete(features, best_attr, axis=1)

        new_attr_names = self.attribute_names.copy()
        new_attr_names.pop(best_attr)

        branch1 = DecisionTree(new_attr_names)
        newfeatures1 = features[np.where(attr_col == 0), :].squeeze(axis=0)
        newtargets1 = targets[np.where(attr_col == 0)]

        branch2 = DecisionTree(new_attr_names)
        newfeatures2 = features[np.where(attr_col == 1), :].squeeze(axis=0) 
        newtargets2 = targets[np.where(attr_col == 1)]

        if newtargets1.size == 0:
            newtargets1 = newtargets2
        elif newtargets2.size == 0:
            newtargets2 = newtargets1

        branch1.fit(newfeatures1, newtargets1)
        branch2.fit(newfeatures2, newtargets2)

        self.tree.branches = [branch1, branch2]
        return

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        def search(example):
            curr = self.tree
            while len(curr.branches) != 0:
                attribute = example[curr.attribute_index]
                example = np.delete(example, curr.attribute_index, 0)
                curr = curr.branches[int(attribute)].tree
            return curr.value

        self._check_input(features)

        preds = np.zeros(features.shape[0])
        for i in range(preds.size):
            preds[i] = search(features[i, :])
        return preds

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    num_examples = features.shape[0]
    num_postar = np.sum(targets)
    num_negtar = num_examples - num_postar
    num_posattr = np.sum(features[:, attribute_index])
    num_negattr = num_examples - num_posattr

    comp_mat = np.zeros((2, 2))
    for f, t in zip(features[:, attribute_index], targets):
        comp_mat[int(f)][int(t)] += 1

    init_ent = -num_postar / num_examples * np.log2(num_postar / num_examples) - num_negtar / num_examples * np.log2(
        num_negtar / num_examples)

    if comp_mat[1, 0] == 0 and comp_mat[1, 1] == 0:
        pos_ent = 0
    elif comp_mat[1, 0] == 0:
        pos_ent = -comp_mat[1, 1] / num_posattr * np.log2(comp_mat[1, 1] / num_posattr)
    elif comp_mat[1, 1] == 0:
        pos_ent = -comp_mat[1, 0] / num_posattr * np.log2(comp_mat[1, 0] / num_posattr)
    else:
        pos_ent = -comp_mat[1, 0] / num_posattr * np.log2(comp_mat[1, 0] / num_posattr) - comp_mat[
            1, 1] / num_posattr * np.log2(comp_mat[1, 1] / num_posattr)

    if comp_mat[0, 0] == 0 and comp_mat[0, 1] == 0:
        neg_ent = 0
    elif comp_mat[0, 0] == 0:
        neg_ent = -comp_mat[0, 1] / num_negattr * np.log2(comp_mat[0, 1] / num_negattr)
    elif comp_mat[0, 1] == 0:
        neg_ent = -comp_mat[0, 0] / num_negattr * np.log2(comp_mat[0, 0] / num_negattr)
    else:
        neg_ent = -comp_mat[0, 0] / num_negattr * np.log2(comp_mat[0, 0] / num_negattr) - comp_mat[
            0, 1] / num_negattr * np.log2(comp_mat[0, 1] / num_negattr)

    gain = init_ent - (num_posattr / num_examples) * pos_ent - (num_negattr / num_examples) * neg_ent
    return gain

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
