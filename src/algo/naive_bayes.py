import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        self.name = "naive bayes"
        self.classes = None
        
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        # get number of samples, features and classes
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        
        # calculate mean, var and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_class = X_train[y_train == c]
            self._mean[idx, :] = X_class.mean(axis=0)
            self._var[idx, :] = X_class.var(axis=0)
            self._priors[idx] = X_class.shape[0] / float(n_samples)
        
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        y_pred = [self._predict(x) for x in X_test]
        return y_pred

    # compute class based on highest posterior for each sample
    def _predict(self, X_test:np.ndarray) -> int:
        posteriors = []
        
        # compute posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, X_test)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    # compute probability density function for each feature
    def _pdf(self, class_idx:int, x:np.ndarray) -> float:
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator