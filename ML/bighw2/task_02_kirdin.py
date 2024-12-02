from typing import SupportsIndex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmps
from sklearn.metrics import classification_report, accuracy_score, f1_score


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

        self.X_train = X

        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for a set of input samples.
        """

        class_labels = np.unique(self.y_train)

        # repeat train and test for them to have the same dimensions
        # this is done in order for them to be subtractable from each other
        test_reshaped = np.repeat(X, self.X_train.shape[0]).reshape(
            [X.shape[0], X.shape[1], self.X_train.shape[0]]
        )
        train_reshaped = (
            np.repeat(self.X_train, X.shape[0])
            .reshape([self.X_train.shape[0], X.shape[1], X.shape[0]])
            .T
        )

        D = np.linalg.norm((test_reshaped - train_reshaped), axis=1)

        # repeat class labels for each unique class
        train_classes = (
            self.y_train.repeat(X.shape[0])
            .reshape([self.y_train.shape[0], X.shape[0]])
            .T
        )
        # sort the distaces in ascending order
        sort_ids = np.argsort(D, axis=1)
        d_sorted = np.dstack(
            [
                np.take_along_axis(D, sort_ids, axis=1),
                np.take_along_axis(train_classes, sort_ids, axis=1),
            ]
        )

        classes_stacked = class_labels.repeat(
            X.shape[0] * self.n_neighbors, axis=0
        ).reshape(class_labels.shape[0], X.shape[0] * self.n_neighbors)

        # count instances of every class in self.n_neigbors nearest data points
        class_counts = np.count_nonzero(
            (
                np.ndarray.flatten(d_sorted[:, : self.n_neighbors, 1])
                == classes_stacked
            ).reshape(
                class_labels.shape[0],
                X.shape[0],
                self.n_neighbors,
            ),
            axis=2,
        )

        # from repeated class columns select those correspoing to
        # the class with most occurances in self.n_neigbors nearest
        # train data points for each test point
        return np.take_along_axis(
            classes_stacked[:, :X.shape[0]],
            np.argmax(class_counts, axis=0, keepdims=True),
            axis=0,
        ).flatten()

    def set_neighbors(self, new_neighbors:int=None):
        '''
        Sets the `n_neigbors` property value.
        '''
        if not new_neighbors is None:
            self.n_neighbors = new_neighbors


def accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Computes the fraction of correctly predicted labels.
    This is a simple yet imperfect measure of classification performance.
    """
    return np.count_nonzero(labels_pred == labels_true) / labels_true.shape[0]


def metric(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Implements the `f1-macro` metric. Computes the number of correct positive, negative and 
    incorrect negative predictions and their `f1` for each class and then averages those scores.
    """
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            f"Non-matching input arguments' shapes: got {labels_true.shape}, {labels_pred.shape}"
        )

    label_classes = np.unique(labels_true)

    score = 0

    for label_class in label_classes:
        tp = np.count_nonzero(
            np.logical_and(labels_pred == label_class, labels_true == label_class)
        )
        fp = np.count_nonzero(
            np.logical_and(labels_pred == label_class, labels_true != label_class)
        )
        fn = np.count_nonzero(
            np.logical_and(labels_pred != label_class, labels_true == label_class)
        )

        score += 2 * tp / (2 * tp + fp + fn)

    return score / label_classes.shape[0]


def plot_classes(
    axis: plt.Axes, x: np.ndarray, y: np.array, cmap: list = None, title: str = None
):
    """
    Draw the provided points on 2D plot and color them according to their class.
    """
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if cmap is None:
        cmap = list(cmps["Set1"].colors)

    classes = np.unique(y)
    for m in range(classes.shape[0]):
        axis.scatter(
            x[y == classes[m], 0],
            x[y == classes[m], 1],
            s=2.5,
            color=cmap[m],
            label=f"class {m+1}",
        )

    axis.legend()
    if isinstance(title, str):
        axis.set_title(title)


if __name__ == "__main__":
    cm = list(cmps["Set1"].colors)
    axes = [plt.figure().add_subplot() for i in range(4)]

    # Fix random seed for reproducibility
    np.random.seed(100)

    # Create synthetic dataset for training and testing
    # Here I decided to try multiclass classification
    means0, covs0 = [1, -1], [[7, 3], [3, 7]]
    x0, y0 = np.random.multivariate_normal(means0, covs0, 190).T

    means1, covs1 = [0, -4], [[0.1, 0.0], [0.0, 25]]
    x1, y1 = np.random.multivariate_normal(means1, covs1, 100).T

    means2, covs2 = [5, -10], [[2, -1], [-1, 2]]
    x2, y2 = np.random.multivariate_normal(means2, covs2, 120).T

    # Convert data to the appropriate format
    data0, labels0 = np.vstack([x0, y0]).T, np.zeros(len(x0))
    data1, labels1 = np.vstack([x1, y1]).T, np.ones(len(x1))
    data2, labels2 = np.vstack([x2, y2]).T, 2 * np.ones(len(x2))

    data = np.vstack([data0, data1, data2])
    labels = np.hstack([labels0, labels1, labels2])
    total_size = data.shape[0]
    print("Dataset shape:", data.shape, labels.shape)

    # Split dataset into 70% train and 30% test
    train_size = int(total_size * 0.7)
    indices = np.random.permutation(total_size)
    X_train, y_train = data[indices][:train_size], labels[indices][:train_size]
    X_test, y_test = data[indices][train_size:], labels[indices][train_size:]
    print("Train/test shapes:", X_train.shape, X_test.shape)

    plot_classes(axes[0], X_test, y_test, cmap=cm, title="Ground truth labels")

    # Create KNN classifier instance
    predictor = KNN(n_neighbors=1)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    
    plot_classes(
        axes[1], X_test, y_pred, cmap=cm, title="Predictions with n_neighbors = 1"
    )

    metrix = [[]]

    for i in range(1, 6):

        # change the amount of neighbors we have
        predictor.set_neighbors(i+1)
        y_pred = predictor.predict(X_test)

        print(f"knn with {i} nearest neghbours:")

        acc = accuracy(y_test, y_pred)
        f1_macro = metric(y_test, y_pred)
        metrix += [[acc, f1_macro]]

        # let's compare implemented accuracy to its sklearn counterpart
        print(f"Accuracy: {acc:.4f} [ours]")
        assert (
            abs(acc - accuracy_score(y_test, y_pred)) < 1e-5
        ), "Implemented accuracy is not the same as sci-kit learn one!"

        # Check classifier performance
        assert (
            accuracy_score(y_test, y_pred) > 190.0 / 410.0
        ), f"This classifier is worse than a constant C={190.0 / 410.0}."

        # Calculate additional metric and compare with library version
        print(f"Additional metric: {f1_macro:.4f} [custom]")
        assert (
            abs(f1_macro - f1_score(y_test, y_pred, average="macro")) < 1e-5
        ), "Custom metric does not match sklearn metric!"

        # Convenient sklearn tool to calculate standard metrics
        print(classification_report(y_test, y_pred))

    metrix = np.array(metrix[:][1:])

    best_n_neighbors = np.int64(np.argmax(metrix[:, 1]) + 1)
    predictor = KNN(n_neighbors=best_n_neighbors)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    plot_classes(
        axes[2],
        X_test,
        y_pred,
        cmap=cm,
        title=f"Best predictions(n_neighbors = {best_n_neighbors})",
    )

    axes[3].scatter(np.arange(1, 6, 1), metrix[:, 0], label="accuracy", c="g")
    axes[3].scatter(np.arange(1, 6, 1), metrix[:, 1], label="f1-macro", c="r")
    axes[3].plot(np.arange(1, 6, 1), metrix[:, 0], c="g")
    axes[3].plot(np.arange(1, 6, 1), metrix[:, 1], c="r")
    axes[3].legend()
    axes[3].grid()
    axes[3].set_xlabel("n_neighbors")
    axes[3].set_ylabel("metric value")
    axes[3].set_xticks([1, 2, 3, 4, 5])
    axes[3].set_title("Metrics as functions of n_neighbors")

    for ax in axes:
        ax.figure.savefig(f"{str(ax.get_title())}.png")

    plt.show()
