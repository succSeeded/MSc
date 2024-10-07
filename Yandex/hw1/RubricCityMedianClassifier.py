import numpy as np
from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.medians = X.groupby(["city", "modified_rubrics"])["average_bill"].median()

    def predict(self, X=None):
        return X.drop(columns=["average_bill"]).merge(
            self.medians,
            how="left",
            left_on=["city", "modified_rubrics"],
            right_index=True,
        )["average_bill"]