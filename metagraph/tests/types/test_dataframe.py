import pytest
from metagraph.plugins.pandas.types import PandasDataFrameType
import pandas as pd


def test_pandas():
    PandasDataFrameType.assert_equal(
        pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"]),
        pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"]),
        {},
        {},
        {},
        {},
    )
    PandasDataFrameType.assert_equal(
        pd.DataFrame([[1, 2, 3.33333333333], [4, 5, 6]], columns=["A", "B", "C"]),
        pd.DataFrame([[1, 2, 3.33333333334], [4, 5, 6]], columns=["A", "B", "C"]),
        {},
        {},
        {},
        {},
    )
    # Order of columns doesn't matter
    PandasDataFrameType.assert_equal(
        pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"]),
        pd.DataFrame([[1, 3, 2], [4, 6, 5]], columns=["A", "C", "B"]),
        {},
        {},
        {},
        {},
    )
    # Index order doesn't matter
    PandasDataFrameType.assert_equal(
        pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=[1, 2]),
        pd.DataFrame([[4, 5, 6], [1, 2, 3]], columns=["A", "B", "C"], index=[2, 1]),
        {},
        {},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        PandasDataFrameType.assert_equal(
            pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"]),
            pd.DataFrame([[1, 2, 3], [4, 5, 7]], columns=["A", "B", "C"]),
            {},
            {},
            {},
            {},
        )
