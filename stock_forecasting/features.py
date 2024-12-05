from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from typing import Tuple

import typer
from loguru import logger
from tqdm import tqdm

from .config import CONFIG
from .utils import waveletSmooth

app = typer.Typer()


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, level=1, wavelet="db1"):
        self.level = level
        self.wavelet = wavelet

        self.prev_x = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.prev_x is None:
            X_transformed = np.copy(X)
            for i in range(X.shape[1]):
                X_transformed[:, i] = waveletSmooth(
                    X[:, i], level=self.level, wavelet=self.wavelet
                )[-len(X) :]

            self.prev_x = X_transformed
            return X_transformed

        else:
            # for the validation we have to do the transform using training data + the current and past validation data
            # i.e. we CAN'T USE all the validation data because we would then look into the future
            temp = np.copy(self.prev_x)
            X_test_WT = np.copy(X)
            for j in range(X.shape[0]):
                # first concatenate train with the latest validation sample
                temp = np.append(temp, np.expand_dims(X[j, :], axis=0), axis=0)
                for i in range(X.shape[1]):
                    X_test_WT[j, i] = waveletSmooth(temp[:, i], level=1)[-1]

            return X_test_WT


def delete_recent(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    # Filter new stocks
    na_count = data.iloc[threshold].isna().groupby(level=0).sum()
    to_remove = na_count[na_count != 0].index

    data = data.drop(columns=to_remove, level=0)
    data.columns = data.columns.remove_unused_levels()
    return data


def calculate_stock_stats(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy(deep=False)

    # Calculate mean daily volume
    for ticker in data.columns.levels[0]:
        data[(ticker, "Dollar_Volume")] = data[(ticker, "Close")] * data[(ticker, "Volume")]

    mean_volume = data.xs("Volume", level="Price", axis=1).mean()
    mean_dollar_volume = data.xs("Dollar_Volume", level="Price", axis=1).mean()

    # Cumulative return (1 year, adjusted for 252 working days)
    for ticker in data.columns.levels[0]:
        # PCT change is scale independent :D
        data[(ticker, "Daily_Return")] = data[(ticker, "Close")].pct_change(fill_method=None)

    annual_return = data.xs("Daily_Return", level="Price", axis=1).mean() * 252

    # Historical volatility (annualized standard deviation)
    annual_volatility = data.xs("Daily_Return", level="Price", axis=1).std() * (252**0.5)

    return pd.DataFrame(
        {
            "mean_volume": mean_volume,
            "mean_dollar_volume": mean_dollar_volume,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
        },
        index=data.columns.levels[0],
    )


def filter_stocks(data: pd.DataFrame, stock_stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    conf = CONFIG["stock_filter"]

    # Filter by liquity
    liquid_assets = stock_stats["mean_volume"] > conf["min_mean_volume"]
    liquid_dollar_assets = stock_stats["mean_dollar_volume"] > conf["min_mean_dollar_volume"]

    # Filter by return and volatility
    performing_assets = stock_stats["annual_return"] > conf["min_annual_return"]
    stable_assets = (stock_stats["annual_volatility"] > conf["min_annual_volatility"]) & (
        stock_stats["annual_volatility"] < conf["max_annual_volatility"]
    )

    filtered = liquid_assets & liquid_dollar_assets & performing_assets & stable_assets

    categories_to_drop = data.columns.levels[0][~filtered]
    data_filtered = data.drop(columns=categories_to_drop, level=0)

    # Update dataframe columns
    data_filtered.columns = data_filtered.columns.remove_unused_levels()

    return data_filtered, categories_to_drop


def calculate_rsi(data: pd.DataFrame) -> pd.Series:
    # Calculate price changes
    delta = data["Target"].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()

    # Calculate the relative strength (RS)
    rs = gain / loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def extract_features(data_all: pd.DataFrame) -> pd.DataFrame:
    data_all = data_all.copy(deep=False)

    # horizons = [2, 5, 60, 250, 1000]
    horizons = [2, 5, 60, 250]

    new_features = {}

    for ticker in data_all.columns.levels[0]:
        # Add target column
        data_all[(ticker, "Target")] = data_all[(ticker, "Close")].pct_change(fill_method=None)

        data = data_all.xs(ticker, level="Ticker", axis=1)

        # Add horizon features
        for horizon in horizons:
            rolling_avg = data["Close"].rolling(horizon).mean()

            # How much the close price has changed compared to previous days
            ratio_column = f"Close_Ratio_{horizon}"
            new_features[(ticker, ratio_column)] = data["Close"] / rolling_avg

            # How much the price went up in the previous days
            trend_column = f"Trend_{horizon}"
            new_features[(ticker, trend_column)] = data["Target"].shift(1).rolling(horizon).sum()

        new_features[(ticker, "EMA")] = data["Target"].ewm(span=12, adjust=False).mean()
        new_features[(ticker, "RSI")] = calculate_rsi(data)

    df = pd.DataFrame(new_features)

    # Set multi-level column names
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Price"])

    return pd.concat(
        [
            df,
            data_all.xs("Target", level="Price", axis=1, drop_level=False),
        ],
        axis=1,
    )


def feature_selection(
    df: pd.DataFrame, min_features_to_select: int, step: int, verbose: bool = False
) -> Tuple[pd.DataFrame, Pipeline]:
    clf = Lasso(alpha=0.1)
    feature_selector = RFE(
        estimator=clf,
        step=step,
        n_features_to_select=min_features_to_select,
        verbose=verbose,
    )

    pipeline = Pipeline(
        [
            ("normalize", RobustScaler()),
            ("feature_selection", feature_selector),
        ],
        verbose=verbose,
    )

    y = df.xs("Target", level="Price", axis=1).shift(-1).dropna()  # Predicting next day's price
    X = df.iloc[:-1]  # Drop the last row of X to align with y

    pipeline.fit(X, y)

    # Filter selected features
    selected_features = pipeline["feature_selection"].support_
    new_df = df.iloc[:, selected_features]

    # Update dataframe columns
    new_df.columns = new_df.columns.remove_unused_levels()

    return new_df, pipeline


def feature_selection2(
    df: pd.DataFrame, min_features_to_select: int, step: int, verbose: bool = False
) -> Tuple[pd.DataFrame, Pipeline]:
    clf = Lasso(alpha=0.1)
    cv = TimeSeriesSplit(n_splits=3)
    feature_selector = RFECV(
        estimator=clf,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=verbose,
    )

    pipeline = Pipeline(
        [
            ("normalize", RobustScaler()),
            ("feature_selection", feature_selector),
        ],
        verbose=verbose,
    )

    y = df.xs("Target", level="Price", axis=1).shift(-1).dropna()  # Predicting next day's price
    X = df.iloc[:-1]  # Drop the last row of X to align with y

    pipeline.fit(X, y)

    # Filter selected features
    selected_features = pipeline["feature_selection"].support_
    new_df = df.iloc[:, selected_features]

    # Update dataframe columns
    new_df.columns = new_df.columns.remove_unused_levels()

    return new_df, pipeline


if __name__ == "__main__":
    app()
