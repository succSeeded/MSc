# coding: utf-8
# coding: utf-8
"""
TL;DR Nearest Neighbors, Synthetic Dataset, Euclidean Distance.

Task: 
    1. Implement a k-nearest neighbors (KNN) classifier for two classes with a configurable number of neighbors. 
    2. Implement the accuracy metric and one other metric of your choice.
    3. Visualize the results.

Grading Criteria:
    The maximum score is 10 points. Your grade will equal the total points earned.
    
    * For KNN implementation: up to 4 points.
        - 3 points are awarded if your code works (passes the corresponding assert).
        - A fourth point is awarded if you use internal numpy functions for implementation.
        - Bonus: An additional point is available (not included in the main 10) for a unique approach in the fit method. See comments in the fit method for details.

    * For accuracy and an additional metric: 1 point each, awarded for passing the asserts.

    * For Matplotlib plots: 1 point for each well-crafted plot. This includes clear points (not overly crowded), readable labels, a plot title, legend, and labeled axes. Partial credit (0.5 points) is given if the plot contains the required data but lacks readability.

    * Significant non-compliance with PEP8: -1 point.

Important: make sure, that your code does not raise any error; otherwise I won't check your asset.

Recommendation:
    Start by examining the code from the `if __name__ == "__main__"` section, then proceed to the accuracy function and the KNN class.
"""

import numpy as np
from typing import SupportsIndex
from sklearn.metrics import classification_report, accuracy_score

class KNN:
    """
    Class implementing the k-nearest neighbors algorithm.
    """

    def __init__(self, n_neighbors: int = 4):
        # Training data: features
        self.X_train = None

        # Training data: class labels
        self.y_train = None

        # Number of nearest neighbors
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: SupportsIndex):
        """
        Fits the KNN model to the training data. 
        In KNN, "fitting" simply involves storing the training dataset.        
        """
        # For the first try, just store the training set in the object's attributes.
        ...
        #  This is fine as is, but real-world implementations use more efficient data structures.
        #  If you implement something innovative, you may earn a bonus point. Note, that the 
        #  external libraries are prohibited.
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predicts labels for a set of input samples.
        """
        # TODO: 1) Compute the Euclidean distance between each sample in X and each sample in self.X_train. 
        ...
        # TODO: 2) For each sample in X, find the self.n_neighbors closest samples in self.X_train and aggregate their labels.
        ...
        # Exercise: Consider the asymptotic complexity of this function. How does it depend on the feature space dimension?

        # Fully vectorized solution: 4 points. Other working solutions: 3 points.
        pass
        

def accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Computes the fraction of correctly predicted labels. 
    This is a simple yet imperfect measure of classification performance.
    """
    return 0.0

def metric(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Implements an additional classification metric. 
    You can choose one we’ve discussed in class or come up with your own.
    """
    return 0.0

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Fix random seed for reproducibility
    np.random.seed(100)

    # Create synthetic dataset for training and testing
    means0, covs0 = [1, -1], [[7, 3], [3, 7]]
    x0, y0 = np.random.multivariate_normal(means0, covs0, 190).T

    means1, covs1 = [0, -4], [[0.1, 0.0], [0.0, 25]]
    x1, y1 = np.random.multivariate_normal(means1, covs1, 100).T

    # Visualize the data (optional)
    # plt.plot(x0, y0, 'o', color='b')
    # plt.plot(x1, y1, 'o', color='r')
    # plt.show()

    # Convert data to the appropriate format
    data0, labels0 = np.vstack([x0, y0]).T, np.zeros(len(x0))
    data1, labels1 = np.vstack([x1, y1]).T, np.ones(len(x1))

    data = np.vstack([data0, data1])
    labels = np.hstack([labels0, labels1])
    total_size = data.shape[0]
    print("Dataset shape:", data.shape, labels.shape)

    # Split dataset into 70% train and 30% test
    train_size = int(total_size * 0.7)
    indices = np.random.permutation(total_size)
    X_train, y_train = data[indices][:train_size], labels[indices][:train_size]
    X_test, y_test = ...
    print("Train/test shapes:", X_train.shape, X_test.shape)

    # TODO: Loop through different values of n_neighbors (1 to 5)

    # Create KNN classifier instance
    predictor = KNN(n_neighbors=3)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    # check that your accuracy is honest :^)
    print("Accuracy: %.4f [ours]" % accuracy(y_test, y_pred))
    assert abs(accuracy_score(y_test, y_pred) - accuracy(y_test, y_pred)) < 1e-5,\
        "Implemented accuracy is not the same as sci-kit learn one!"
    
    # Check classifier performance
    assert accuracy_score(y_test, y_pred) > 190. / 290.\
        "Your classifier is worse than the constant !"

    # Calculate additional metric and compare with library version
    print("Additional metric: %.4f [custom]" % metric(y_test, y_pred))
    assert abs(metric(y_test, y_pred) - ...) < 1e-5, \
        "Custom metric does not match sklearn metric!"

    # Convenient sklearn tool to calculate standard metrics
    print(classification_report(y_test, y_pred))

    # Matplotlib Exercise:
    # Generate three plots for the test set: 
    # - Ground truth labels
    # - Predictions with n_neighbors = 1
    # - Predictions with the best n_neighbors in the range 1...5
    
    # Each plot should include the training data points with appropriate colors, 
    # (hint: using transparency or small markers to avoid covering test points).
    
    # Save plots !!to current folder!! using matplotlib's `savefig`.

    # Fourth plot: Plot metrics as functions of n_neighbors.
    # - Show both metrics on one graph with distinct colors and a legend.
    # - If the scales differ, use two vertical axes.