import random as rnd
import numpy as np

def fit(X, y, max_passes=1,  C=1.0, epsilon=0.001):

    X = np.array(X)
    # initialize alphas
    alphas = np.zeros((len(X)))
    passes = 0

    while passes < max_passes:
        passes += 1
        alpha_prev = np.copy(alphas)
        for i in range(0, int(len(alphas))):

            # weights and bias
            weights = np.dot(X.T, np.multiply(alphas, y))
            bias = np.mean(y - np.dot(weights.T, X.T))

            # Calculate E_i
            x_i = X[i, :]
            y_i = y[i]

            E_i = E(x_i, y_i, weights, bias)

            # Select j!=i
            j = generate_number(0, len(alphas)-1, i)
            x_j = X[j, :]
            y_j = y[j]

            # Calculate E_j
            E_j = E(x_j, y_j, weights, bias)

            # save old alphas
            alpha_old_i = alphas[i]
            alpha_old_j = alphas[j]

            # Compute L and H
            L = Lu(C, alphas[i], alphas[j], y_i, y_j)
            H = Hu(C, alphas[i], alphas[j], y_i, y_j)

            # Compute η
            k_ij = kernel_quadratic(x_j, x_j) + kernel_quadratic(x_i, x_i) - 2 * kernel_quadratic(x_j, x_j)

            if k_ij == 0 or L == H:
                continue

            # compute alpha j
            alphas[j] = alpha_old_j + y[j] * (E_i - E_j) / k_ij

            if (alphas[j] > H):
                alphas[j] = H
            elif (alphas[j] < L):
                alphas[j] = L

            # compute alpha i
            alphas[i] = alpha_old_i + y_i*y_j * (alpha_old_j - alphas[j])


        # Check convergence
        diff = np.linalg.norm(alphas - alpha_prev)
        if diff < epsilon:
            break

    bias = np.mean(y - np.dot(weights.T, X.T))
    return weights, bias


def generate_number(a, b, j):
    c = 0
    for i in range(0, 10000):
        c = rnd.randint(a, b)
        if (c != j):
            return c
    return c


def predict(X, weights, bias):
    return h(X, weights, bias)

def Lu(C, alpha_j, alpha_i, y_j, y_i):
    if (y_i != y_j):
        return max(0, alpha_j - alpha_i)
    else:
        return max(0, alpha_i + alpha_j - C)

def Hu(C, alpha_j, alpha_i, y_j, y_i):
    if(y_i != y_j):
        return min(C, C - alpha_i + alpha_j)
    else:
        return min(C, alpha_i + alpha_j)

def h(X, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)

def E(x_i, y_i, w, b):
    return h(x_i, w, b) - y_i

def kernel_quadratic( x1, x2):
        return (np.dot(x1, x2.T) ** 2)