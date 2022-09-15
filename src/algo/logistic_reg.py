import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.name = "logistic regression"
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        # init weights and bias
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for itr in range(self.n_iters):
            # get predictions
            y_linear_preds = np.dot(X_train, self.weights) + self.bias
            y_preds = sigmoid(y_linear_preds)
            
            # gradient descent
            # compute gradients of weights and bias
            dweights = (1/n_samples) * np.dot(X_train.T, (y_preds - y_train))
            dbias = (1/n_samples) * np.sum(y_preds - y_train)
            
            # update weights and bias
            self.weights = self.weights - self.lr * dweights
            self.bias = self.bias - self.lr * dbias
    
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        # get predictions
        y_linear_preds = np.dot(X_test, self.weights) + self.bias
        y_sigm_preds = sigmoid(y_linear_preds)
        
        # convert to class predictions
        y_preds = [0 if y <= 0.5 else 1 for y in y_sigm_preds]
        return y_preds
        
# define sigmoid function for calculation of classification probabilities
def sigmoid(x:int) -> float:
    return 1/(1+np.exp(-x))