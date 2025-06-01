from typing import Sequence

def fit(X: Sequence[float], Y: Sequence[float]):
    # input validation
    if len(X) != len(Y) or len(set(X)) == 1:
        raise ValueError('X and Y must have the same length and X must contain at least two distinct values.')
    
    n = len(X) # number of samples
    X_mean = sum(X) / n # X sample mean
    Y_mean = sum(Y) / n # Y sample mean

    m = (sum(x * y for x, y in zip(X, Y)) - (n * X_mean * Y_mean)) / (sum(x ** 2 for x in X) - (n * (X_mean ** 2))) # solution for slope when minimizing MSE
    b = Y_mean - (m * X_mean) # solution for y-intercept when minimizing MSE

    return (m, b)

def main():
    X = [1, 2, 3]
    Y = [2, 3, 5]
    m, b = fit(X, Y)
    print("m =", m)  # should be 1.5
    print("b =", b)  # should be approximately 0.3333

if __name__ == '__main__':
    main()