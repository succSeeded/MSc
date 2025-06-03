def cosine_dist(x1: npt.ArrayLike, x2: npt.ArrayLike) -> npt.DTypeLike:
    return x1.dot(x2) / np.sqrt(np.power(x1, 2).sum()) / np.sqrt(np.power(x2, 2).sum())


class ObliviousTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree regressor, but symmetric! All child nodes are such that they split over the same feature and threshold.
    """

    SPLIT_CRITERIA = {
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    }

    def __init__(
        self,
        max_depth: int = None,
        min_samples_leaf: int = None,
        criterion: str = None,
    ) -> None:
        if criterion is None:
            self.criterion = ObliviousTreeRegressor.SPLIT_CRITERIA["mse"]
        else:
            self.criterion = ObliviousTreeRegressor.SPLIT_CRITERIA[criterion.lower()]
        self.max_depth = 4 if max_depth is None else max_depth
        self.min_samples_leaf = 1 if min_samples_leaf is None else min_samples_leaf

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        features: list = None,
    ) -> None:
        """Fit the decision tree to passed data."""

        if features is None:
            self.features_ = [f"x_{i}" for i in range(X.shape[1])]
        else:
            self.features_ = features

        self.tree_ = ObliviousTreeRegressor.build_tree_(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
        )

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict class labels for given data."""
        ans = np.zeros((X.shape[0],))
        for i in range(ans.shape[0]):
            node = self.tree_
            while not node["is_leaf"]:
                # Get the index a the feature on which the split was performed
                feature_idx = self.features_.index(node["feature"])
                if X[i, feature_idx] <= node["threshold"]:
                    node = node["children"]["lower"]
                else:
                    node = node["children"]["upper"]

            ans[i] = node["value"]
        return ans

    def get_n_leaves(self) -> np.int64:
        return np.power(2, self.tree_["depth"], dtype=np.int64)

    # =============================================================================
    # Tree construction
    # =============================================================================

    def build_tree_(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        init_guess: npt.ArrayLike,
        max_depth: int,
        min_samples_leaf: int,
        criterion: Callable,
    ) -> Dict[str, Any]:
        """Recursively build an oblivious regression tree."""

        tree = {
            "depth": 0,
            "leaf_values": np.zeros(1),
            "split_feature": [],
            "split_threshold": [],
        }

        predictions = init_guess
        residuals = y - init_guess
        remaining_features = list(range(X.shape[1]))

        # Depth must be less than or equal to 31 so that there is no integer overflow
        while tree["depth"] <= min(max_depth, 31):

            tree["depth"] += 1

            # Terminate if there are no more features to split on
            if len(tree["split_feature"]) == X.shape[1]:
                break

            best_feature, threshold = ObliviousTreeRegressor.select_best_feature_(
                X,
                y,
                residuals,
                remaining_features,
                tree["split_feature"],
                tree["split_threshold"],
            )
            tree["split_feature"] += [best_feature]
            tree["split_threshold"] += [threshold]
            remaining_features = [
                feature for feature in remaining_features if feature != best_feature
            ]

            # Find the current splits
            res = ObliviousTreeRegressor.find_outputs_(
                X, residuals, tree["split_feature"], tree["split_threshold"]
            )
            predictions = res["tree_outputs"]
            residuals = y - predictions
            tree["leaf_values"] = res["leaf_values"]

            # splits = np.ones_like(y)
            # for split_idx, (feature_idx, threshold) in enumerate(zip(tree["split_feature"], tree["split_threshold"])):
            #     splits = np.where(X[:, feature_idx] <= threshold, splits + np.power(2, split_idx), splits)

            # # Terminate if one of the splits does not contain enough elements to split
            # values, counts = np.unique(splits, return_counts=True)
            # if np.any(counts == 1):
            #     break

            # curr_guess = np.zeros_like(y)
            # for value, count in zip(values, counts):
            #     mask = (splits == value)

            #     # Predictions are the cumulative means of residuals in the corresponding leaves
            #     leaf_residuals = tree["residuals"][mask]
            #     leaf_preds = np.zeros_like(leaf_residuals)
            #     for i in range(1, leaf_residuals.shape[0]):
            #         leaf_preds[i] = (leaf_residuals[i-1] + leaf_preds[i-1]) / i

            #     curr_guess = np.where(mask, leaf_preds, curr_guess)

            # tree["predictions"] = curr_guess

        return tree

    def find_outputs_(
        X: npt.ArrayLike,
        residuals: npt.ArrayLike,
        split_features: list,
        split_thresholds: list,
    ) -> Dict:

        splits = np.ones_like(residuals)
        for split_idx, (feature_idx, threshold) in enumerate(
            zip(split_features, split_thresholds)
        ):
            splits = np.where(
                X[:, feature_idx] <= threshold, splits + np.power(2, split_idx), splits
            )

        values, counts = np.unique(splits, return_counts=True)
        ans = {"tree_outputs": np.zeros_like(y), "leaf_values": np.zeros_like(values)}

        for leaf_id, leaf_name in enumerate(values):
            mask = splits == leaf_name

            # Predictions are the cumulative means of residuals in the corresponding leaves
            leaf_cumresiduals = np.cumsum(residuals[mask])
            leaf_preds = np.zeros_like(leaf_residuals)
            for i in range(1, leaf_cumresiduals.shape[0]):
                leaf_preds[i] = leaf_cumresiduals[i - 1] / i

            ans["leaf_values"][leaf_id] = (
                leaf_cumresiduals[-1] / leaf_cumresiduals.shape[0]
            )  # essentially the mean of all residuals on the leaf of choice
            ans["tree_outputs"] = np.where(mask, leaf_preds, ans["tree_outputs"])
        return ans

    def select_best_feature_(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        residuals: npt.ArrayLike,
        remaining_features: list,
        split_features: list,
        split_thresholds: list,
    ) -> list:
        """Finds a feature and threshold that produce outputs closest to residuals in terms of cosine similarity."""
        best_split_scores = []
        best_thresholds = []

        for feature_idx in remaining_features:
            uniques = np.unique(X[:, feature_idx])

            # Splits are not done if all the feature values are the same
            if uniques.shape[0] == 1:
                return {"value": np.inf, "threshold": None}

            thresholds = [
                0.5 * (curr + prev) for prev, curr in zip(uniques, uniques[1:])
            ]
            split_scores = []

            for theta in thresholds:

                split = ObliviousTreeRegressor.find_outputs_(
                    X,
                    residuals,
                    split_features + [feature_idx],
                    split_thresholds + [theta],
                )
                split_scores += [cosine_dist(residues_pred["tree_outputs"], residuals)]

            best_split_id = np.argmax(split_scores)
            best_split_scores += [split_scores[best_split_id]]
            best_thresholds += [thresholds[best_split_id]]

        best_feature_idx = np.argmax(best_split_scores)
        return [best_feature_idx, best_thresholds[best_feature_idx]]

class CatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator: object = None, n_estimators: int = None):
        self.n_estimators = 50 if n_estimators is None else n_estimators
        self.estimator = DecisionTreeClassifier if estimator is None else estimator
        self.classifier_weights_ = np.zeros(self.n_estimators)
        self.classifiers_ = [self.estimator for i in range(self.n_estimators)]  

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Incorrect input array shape: {X.shape} and {y.shape}")

        n_samples = X.shape[0]

