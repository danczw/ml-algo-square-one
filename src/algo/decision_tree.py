from collections import Counter
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.name = "decision tree"
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        # check feature size
        self.n_features = X_train.shape[1] if not self.n_features \
            else min(self.n_features, X_train.shape[1])
        # get best split
        self.root = self._grow_tree(X_train, y_train)
            
    def _grow_tree(
        self,
        X_train:np.ndarray,
        y_train:np.ndarray,
        depth:int=0
    ) -> Node:
        n_samples, n_features = X_train.shape
        n_labels = len(np.unique(y_train))
        
        # check stopping criteria
        if (depth >= self.max_depth or n_labels==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y_train)
            return Node(value=leaf_value)
        
        # select random features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # find best split
        best_thresh, best_feature = self._best_split(X_train, y_train, feat_idxs)
        
        # create child nodes
        left_idxs, right_idxs = self._split(X_train[:, best_feature], best_thresh)
        
        # grow tree recursively
        grow_left = self._grow_tree(X_train[left_idxs, :], y_train[left_idxs], depth+1)
        grow_right = self._grow_tree(X_train[right_idxs, :], y_train[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, grow_left, grow_right)
    
    # return most common label in a list
    def _most_common_label(self, y_train:np.ndarray) -> int:
        counter = Counter(y_train)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    # compute best split for max information gain
    def _best_split(
        self,
        X_train:np.ndarray,
        y_train:np.ndarray,
        feat_idxs:list
    ) -> tuple:
        best_gain = -1
        split_threshold, split_idx = None, None
        
        # iterate over all features
        for feat_idx in feat_idxs:
            X_train_column = X_train[:, feat_idx]
            thresholds = np.unique(X_train_column)
            
            # test all possible thresholds
            for th in thresholds:
                gain = self._information_gain(y_train, X_train_column, th)
                
                # update best gain and threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = th
        
        return split_threshold, split_idx

    # information gained through split based on entropy
    def _information_gain(
        self,
        y_train:np.ndarray,
        X_train_column:np.ndarray,
        split_threshold:float
    ) -> float:
        # get parent entropy
        parent_entropy = self._entropy(y_train)
        
        # create children
        left_idxs, right_idxs = self._split(X_train_column, split_threshold)
        
        # check for 0 information gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # compute weighted avg. entropy of children
        n = len(y_train)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y_train[left_idxs]), self._entropy(y_train[right_idxs])
        child_entropy = (n_left/n) * e_left + (n_right/n) * e_right
        
        # compute information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    # compute entropy as a measure of impurity of a node
    def _entropy(self, y_train:np.ndarray) -> float:
        # array count of unique values
        hist = np. bincount(y_train)
        
        # array normalization
        ps = hist / len(y_train)
        
        # compute entropy
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    # split data based on threshold and flatten
    def _split(self, X_train_column:np.ndarray, split_threshold:float) -> tuple:
        left_idx = np.argwhere(X_train_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_train_column > split_threshold).flatten()
        
        return left_idx, right_idx
    
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        return np.array([self._traverse_tree(x, self.root) for x in X_test])
    
    # traverse tree to make prediction
    def _traverse_tree(self, x:np.ndarray, node:Node) -> int:
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)