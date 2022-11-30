import numpy as np

class Perceptron:
    def __init__(self, learning_rate:float=0.001, n_iters:int=1000):
        self.name = "perceptron"
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        n_samples, n_features = X_train.shape
        
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y_train_ = np.where(y_train > 0, 1, 0)
        
        # learn weights and bias
        for iter in range(self.n_iters):
            for idx, x_i in enumerate(X_train):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y_train_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    # unit step wise activation function
    def _unit_step_func(self, x:np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)
    
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._unit_step_func(linear_output)
        
        return y_predicted