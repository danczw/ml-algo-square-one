import numpy as np


class SVM():
    def __init__(
        self,
        learning_rate:float=0.001,
        lambda_param:float=0.01,
        n_iters:int=1000
    ) -> None:
        self.name = "Support Vector Machine"
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        n_samples, n_features = X_train.shape

        # set labels to 1 or -1
        y_ = np.where(y_train <= 0, -1, 1)

        # init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # learn weights and bias
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X_train):
                # calulate condition: gradient
                condition = y_[idx] * \
                    (np.dot(x_i, self.weights) - self.bias) >= 1
                
                # update rules based on conidtion
                if condition:
                    self.weights -= self.learning_rate * \
                        (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * \
                        self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        # calculate approximation
        approx = np.dot(X_test, self.weights) - self.bias
        return np.sign(approx)