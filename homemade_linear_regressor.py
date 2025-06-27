import numpy as np
from numpy.typing import ArrayLike

class LinearRegression():
    def __init__(self):
        self.W = None

    @staticmethod
    def _validate(A: ArrayLike, name: str):
        try:
            A = np.asarray(A, dtype=float) # convert to numpy array
        except(ValueError, TypeError): # catch unexpected data types or ragged arrays
            raise ValueError(f'{name} must be a numeric and non-ragged sequence, got {A!r}.')

        # reshape to matrix form
        if A.ndim == 0:
            A = A.reshape(1, 1)
        elif A.ndim == 1:
            A = A.reshape(-1, 1)
        elif A.ndim == 2:
            pass # already properly shaped
        else:
            raise ValueError(f'{name} must be at most 2-dimensional, got {name}.ndim = {A.ndim}.')
        
        # check if array has at least one value
        if 0 in A.shape:
            raise ValueError(f'{name} cannot be empty, got {A!r}.')
        
        return A
    
    def _print_W(self):
        # ANSI color codes
        GREEN = '\033[1;32m'
        WHITE = '\033[1m'
        RESET = '\033[0m'
    
        bias = self.W[0, 0]
        coefficients = self.W[1:, 0]
        
        print(f'{GREEN}Linear Regression Successfully Fitted{RESET}\n')
        print(f'{WHITE}Bias:{RESET} {bias}')
        print(f'{WHITE}Coefficients:{RESET} {coefficients}')

    def fit(self, X: ArrayLike, Y: ArrayLike):
        try:
            # input validation and reshaping
            X, Y = self._validate(X, 'X'), self._validate(Y, 'Y')
            assert X.shape[0] == Y.shape[0], 'X and Y must have the same length.'

            # prepend 1 to each vector in X so the bias term is learned as part of the weight vector
            ones = np.ones((X.shape[0], 1))
            X_aug = np.hstack((ones, X))

            self.W = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ Y) # compute weight vector 
            self._print_W()
        except Exception as e:
            raise RuntimeError(f'Failed to fit: {e}') from None