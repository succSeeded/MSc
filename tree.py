from collections import Counter
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import numpy.typing as npt

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree regressor, something to build gradient boosting algorithms off of.
    """

    SPLIT_CRITERIA = {
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    }

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        criterion: str = None,
    ) -> None:
        if criterion is None:
            self.criterion = DecisionTreeRegressor.SPLIT_CRITERIA["mse"]
        else:
            self.criterion = DecisionTreeRegressor.SPLIT_CRITERIA[criterion.lower()]
        self.max_depth = -1 if max_depth is None else max_depth
        self.min_samples_split = 2 if min_samples_split is None else min_samples_split
        self.min_samples_leaf = 1 if min_samples_leaf is None else min_samples_leaf

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        features: list = None,
        cat_features: list = None,
    ) -> None:
        """Fit the decision tree to passed data."""

        if features is None:
            self.features_ = [f"x_{i}" for i in range(X.shape[1])]
        else:
            self.features_ = features

        if cat_features is None:
            self.cat_features_ = []
        else:
            self.cat_features_ = cat_features

        self.tree_ = DecisionTreeRegressor.build_tree_(
            X,
            y,
            self.features_,
            depth=0,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            cat_features=self.cat_features_,
            criterion=self.criterion,
        )

        self.n_leaves_ = DecisionTreeRegressor.count_leaves_(self.tree_)

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict class labels for given data."""
        ans = np.zeros((X.shape[0],))
        for i in range(ans.shape[0]):
            node = self.tree_
            while not node["is_leaf"]:
                # Get the index a the feature on which the split was performed
                feature_idx = self.features_.index(node["feature"])

                if node["feature"] in self.cat_features_:
                    # If the category has not appeared in training set, the tree
                    # traversal is terminated and the current node value is used
                    if node["children"].get(X[i, feature_idx], False):
                        node = node["children"].get(X[i, feature_idx], False)
                    else:
                        break
                else:
                    if X[i, feature_idx] <= node["threshold"]:
                        node = node["children"]["lower"]
                    else:
                        node = node["children"]["upper"]

            ans[i] = node["value"]
        return ans

    def get_n_leaves(self):
        return self.n_leaves_

    # =============================================================================
    # Tree construction
    # =============================================================================

    def build_tree_(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        features: list,
        depth: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        cat_features: list,
        criterion: Callable,
    ) -> Dict[str, Any]:
        """Recursively build a regression tree."""

        default_value = np.mean(y)

        # Terminate if there are no more features to split on
        if X.shape[1] == 0 or len(features) == 0:
            return {"value": default_value, "depth": depth, "is_leaf": True}

        # Terminate if all the targets are duplicates of eachother
        if np.unique(y).shape[0] == 1:
            return {"value": default_value, "depth": depth, "is_leaf": True}

        # Terminate if all the datapoints are duplicates of eachother
        if np.unique(X, axis=0).shape[0] == 1:
            return {"value": default_value, "depth": depth, "is_leaf": True}

        # Terminate if node does not contain enough elements to split
        if y.shape[0] < min_samples_split:
            return {"value": default_value, "depth": depth, "is_leaf": True}

        # Terminate if max tree depth is reached
        if depth == max_depth:
            return {"value": default_value, "depth": depth, "is_leaf": True}

        best_id, best_feature, threshold = DecisionTreeRegressor.select_best_feature_(
            X, y, features, cat_features, criterion
        )
        new_features = [feature for feature in features if feature != best_feature]

        tree = {
            "depth": depth,
            "feature": best_feature,
            "is_leaf": False,
            "value": default_value,
            "children": {},
        }

        if best_feature in cat_features:
            categories = np.unique(X[:, best_id])

            for category in categories:

                mask = X[:, best_id] == category

                X_sub, y_sub = (
                    np.delete(X, best_id, axis=1)[mask],
                    y[mask],
                )

                # If there is not enough samples to make a leaf the split is aborted and
                # the node is considered a leaf
                if len(y_sub) < min_samples_leaf:
                    tree["is_leaf"] = True
                    tree["children"] = {}
                    return tree

                tree["children"][category] = DecisionTreeRegressor.build_tree_(
                    X_sub,
                    y_sub,
                    new_features,
                    depth=depth + 1,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    cat_features=cat_features,
                    criterion=criterion,
                )
        else:
            mask_left = X[:, best_id] <= threshold
            mask_right = X[:, best_id] > threshold

            tree["threshold"] = threshold

            # If there is not enough samples to make a leaf the split is aborted and
            # the node is considered a leaf
            if (
                mask_left.sum() < min_samples_leaf
                or mask_right.sum() < min_samples_leaf
            ):
                tree["is_leaf"] = True
                return tree

            X_sub = np.delete(X, best_id, axis=1)

            tree["children"]["lower"] = DecisionTreeRegressor.build_tree_(
                X_sub[mask_left],
                y[mask_left],
                new_features,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                cat_features=cat_features,
                criterion=criterion,
            )

            tree["children"]["upper"] = DecisionTreeRegressor.build_tree_(
                X_sub[mask_right],
                y[mask_right],
                new_features,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                cat_features=cat_features,
                criterion=criterion,
            )
        return tree

    def feature_score_(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        feature_idx: int,
        is_cat_feature: bool,
        criterion: Callable,
    ) -> float:
        """Calculate score for a given feature."""

        if is_cat_feature is None or is_cat_feature == False:

            uniques = np.unique(X[:, feature_idx])

            # Splits are not done if all the feature values are the same
            if uniques.shape[0] == 1:
                return {"value": np.inf, "threshold": None}

            thresholds = [
                0.5 * (curr + prev) for prev, curr in zip(uniques, uniques[1:])
            ]
            split_scores = []

            for theta in thresholds:

                mask_left = X[:, feature_idx] <= theta
                mask_right = X[:, feature_idx] > theta

                # Elements lower and higher than threshold are compared to their respective means
                y_pred = np.where(
                    mask_left, np.mean(y[mask_left]), np.mean(y[mask_right])
                )
                split_scores += [criterion(y_pred, y)]

            best_split_id = np.argmin(split_scores)
            best_threshold = thresholds[best_split_id]
            return {"value": split_scores[best_split_id], "threshold": best_threshold}

        else:
            categories = np.unique(X[:, feature_idx])
            y_pred = np.zeros_like(y)

            for category in categories:

                mask = X[:, feature_idx] == category
                # Elements in each(surviving!) category are compared to their respective means
                y_pred = np.where(mask, np.mean(y[mask]), y_pred)

            score = criterion(y_pred, y)
            return {"value": score, "threshold": None}

    def select_best_feature_(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        features: list,
        cat_features: list,
        criterion: Callable,
    ) -> list:
        """Select the feature with the highest information gain."""
        scores = [
            DecisionTreeRegressor.feature_score_(
                X, y, i, feature in cat_features, criterion
            )
            for i, feature in enumerate(features)
        ]

        best_idx = np.argmin([score["value"] for score in scores])
        return [best_idx, features[best_idx], scores[best_idx]["threshold"]]

    # =============================================================================
    # Tree pruning
    # =============================================================================

    def count_leaves_(tree: Dict) -> int:
        """Count the number of leaf nodes in a (sub)tree."""
        if tree["is_leaf"]:
            return 1

        total_leaves = 0
        for child in tree["children"]:
            total_leaves += DecisionTreeRegressor.count_leaves_(tree["children"][child])

        return total_leaves
