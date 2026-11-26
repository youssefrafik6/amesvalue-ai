"""Fold-local feature encoding and scaling."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from amesvalue.data import CATEGORICAL, NUMERIC


def preprocessor(*, encode_reference=False):
    """Treatment contrasts for linear models and full encoding for trees."""
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERIC,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first" if encode_reference else None,
                        dtype=np.float64,
                    ),
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
