import numpy as np


def entropy(y: np.ndarray) -> float:
    """Calculate the entropy of a target array.""" 
    _, counts = np.unique(y, return_counts=True)
    probs = counts / y.shape[0]
    return -np.sum(probs * np.log(probs + 1e-10))


def information_gain(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    """Calculate the information gain for a given feature."""
    parent_entropy = entropy(y)
    conditional_entropy = 0

    for value, count in zip(*np.unique(X[:, feature_idx], return_counts=True)):
        p = count / y.shape[0]
        conditional_entropy += p * entropy(y[X[:, feature_idx] == value])

    return parent_entropy - conditional_entropy


def id3_algorithm(X: np.ndarray, y: np.ndarray, features: list) -> dict[str, Any]:
    """Recursively build the ID3 decision tree."""
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]

    # Base case: all samples same class
    if len(classes) == 1 or not features:
        return {'class': classes[0], 'majority_class': classes[0]}

    best_feature = ...
    feature_values = ...
    new_features = ...

    tree = {
        'feature': best_feature,
        'majority_class': majority_class,
        'children': {}
    }

    for value in feature_values:
        mask = ...
        X_sub, y_sub = X[mask], y[mask]
        if len(y_sub) == 0:
            tree['children'][value] = {'class': majority_class, 'majority_class': majority_class}
        else:
            tree['children'][value] = id3_algorithm(X_sub, y_sub, new_features)

    return tree


if __name__=="__main__":
    print(information_gain(X=np.array([[0, 1],[0, 0]]), y=np.array([1,0]), feature_idx=1))
