"""Grouped information-criterion selection and sale-price regressors."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression

from amesvalue.data import CATEGORICAL, NUMERIC
from amesvalue.features import preprocessor


class LinearBenchmark(RegressorMixin, BaseEstimator):
    """Fit log or dollar regression with optional term selection and retransformation."""

    def __init__(self, method="full", transform="smearing", estimator=None):
        self.method = method
        self.transform = transform
        self.estimator = estimator

    def fit(self, X, y):
        self.preprocessor_ = preprocessor(encode_reference=True)
        design = self.preprocessor_.fit_transform(X)
        encoder = self.preprocessor_.named_transformers_["categorical"][-1]
        names = self.preprocessor_.get_feature_names_out()
        groups = {name: [i] for i, name in enumerate(NUMERIC)}
        offset = len(NUMERIC)
        for feature, levels in zip(CATEGORICAL, encoder.categories_, strict=True):
            groups[feature] = list(range(offset, offset + len(levels) - 1))
            offset += len(levels) - 1
        target = np.asarray(y) if self.transform == "raw" else np.log(y)
        self.indices_ = list(range(design.shape[1]))
        self.interaction_indices_ = []
        if self.method in {"aic", "bic"}:
            self.indices_, self.terms_ = select(design, target, groups, self.method)
        if self.method == "interaction":
            self.interaction_indices_ = groups["neighborhood"]
        final = self._design(design)
        baseline = (
            LinearRegression() if self.estimator is None else clone(self.estimator)
        )
        self.model_ = baseline.fit(final, target)
        errors = target - self.model_.predict(final)
        self.factor_ = {
            "smearing": lambda: np.mean(np.exp(errors)),
            "normal": lambda: np.exp(np.var(errors, ddof=1) / 2),
        }.get(self.transform, lambda: 1.0)()
        self.column_names_ = list(names[self.indices_]) + [
            f"overall_quality:{names[i]}" for i in self.interaction_indices_
        ]
        return self

    def _design(self, matrix):
        interactions = matrix[:, [0]] * matrix[:, self.interaction_indices_]
        selected = matrix[:, self.indices_]
        if selected.shape[1] == 0:
            return np.zeros((matrix.shape[0], 1))
        return np.column_stack([selected, interactions])

    def design(self, X):
        return self._design(self.preprocessor_.transform(X))

    def predict(self, X):
        prediction = self.model_.predict(self.design(X))
        if self.transform == "raw":
            return prediction
        return np.exp(prediction) * self.factor_


def select(design, target, groups, method):
    """Search additions/deletions by whole terms, using fitted matrix rank."""
    active = frozenset(groups) if method == "bic" else frozenset()
    n = len(target)
    penalty = np.log(n) if method == "bic" else 2
    scores = {}

    def columns(terms):
        return [i for term, indices in groups.items() if term in terms for i in indices]

    def score(terms):
        if terms not in scores:
            matrix = np.column_stack([np.ones(n), design[:, columns(terms)]])
            coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            error = target - matrix @ coef
            scores[terms] = n * np.log(error @ error / n) + penalty * rank
        return scores[terms]

    while True:
        candidates = active if method == "bic" else groups
        best = min(
            [active, *(active ^ {term} for term in sorted(candidates))], key=score
        )
        if score(best) >= score(active) - 1e-8:
            return columns(active), sorted(active)
        active = best
