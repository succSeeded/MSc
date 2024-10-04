from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.medians = X.groupby(["city", "modified_rubrics"])["average_bill"].median()

    def predict(self, X=None):
        return X.merge(self.medians, how="left", on=["city", "modified_rubrics"])