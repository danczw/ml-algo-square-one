from collections import Counter
import numpy as np

# dynamic entry point import
try:
    from decision_tree import DecisionTree
except:
    from algo.decision_tree import DecisionTree

class RandomForest:
    def __init__(
        self,
        n_trees:int=10,
        max_depth:int=10,
        min_samples_split:int=2,
        n_features:int=None
    ) -> None:
        self.name = "random forest"
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.forest = []
        
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        # create n trees trained on random subsample of data
        for tree in range(self.n_trees):
            # init new tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            
            # create subsample
            X_sample, y_sample = self._bootstrap_sample(X_train, y_train)
            
            # fit tree and append to forest
            tree.fit(X_train, y_train)
            self.forest.append(tree)
    
    # create subsample of train data by random
    def _bootstrap_sample(self, X_train:np.ndarray, y_train:np.ndarray) -> tuple:
        n_samples = X_train.shape[0]
        indeces = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = X_train[indeces]
        y_sample = y_train[indeces]
        return X_sample, y_sample
        
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        # get list of lists: grouped by prediction by tree
        y_preds_treewise = np.array([tree.predict(X_test) for tree in self.forest])
        
        # transform to list of lists: grouped by prediction for sample
        y_preds_samplewise = np.swapaxes(y_preds_treewise, 0, 1)
        
        # compute most common label for each sample prediction
        y_preds = np.array(
            [self._most_common_label(preds) for preds in y_preds_samplewise]
        )
        
        return y_preds
        
    # return most common label in a list
    def _most_common_label(self, y_train:np.ndarray) -> int:
        counter = Counter(y_train)
        most_common = counter.most_common(1)[0][0]
        return most_common