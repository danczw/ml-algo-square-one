import numpy as np

class PCA:
    def __init__(self, n_components:int) -> None:
        self.name = "pca"
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray) -> None:
        # mean centering
        self.mean = np.mean(X_train, axis=0)
        X_train_centered = X_train - self.mean
        
        # compute covariance, Attention: np.cov() needs samples as columns
        cov = np.cov(X_train_centered.T)
        
        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        
        # sort eigenvectors and eigenvalues by eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # store first n eigenvectors
        self.components = eigenvectors[:self.n_components]
    
    # transform x using test data
    def transform(self, X_test:np.ndarray) -> list:
        X_test_centered = X_test - self.mean
        return np.dot(X_test_centered, self.components.T)