import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1) # ensure y is a column vector
        assert X.shape[0] == y.shape[0], 'X and y must have the same length.' # input validation

        # prepend 1 to each vector in X so the bias term is learned as part of the weight vector
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack((ones, X))

        self.w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y) # compute weight vector 

        # todo: add colorful terminal logging to show model is trained