from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

import typer
from loguru import logger
from tqdm import tqdm

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, CONFIG, PROCESSED_DATA_DIR
from .utils import wavelet_transform, waveletSmooth

MIN_MEAN_VOLUME = 500_000
MIN_MEAN_DOLLAR_VOLUME = 10_000_000

MIN_ANNUAL_RETURN = 0.05
MIN_ANNUAL_VOLATILITY = 0.1
MAX_ANNUAL_VOLATILITY = 0.5

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


@app.command()
def filter_stocks(
    input_path: Path = RAW_DATA_DIR / "raw.pkl",
    output_path: Path = RAW_DATA_DIR / "raw_filtered.pkl",
):
    data: pd.DataFrame = pd.read_pickle(input_path)

    # Calculate mean daily volume
    for ticker in data.columns.levels[0]:
        data[(ticker, "Dollar_Volume")] = data[(ticker, "Close")] * data[(ticker, "Volume")]

    mean_volume = data.xs("Volume", level="Price", axis=1).mean()
    mean_dollar_volume = data.xs("Dollar_Volume", level="Price", axis=1).mean()

    # Filter by liquity
    liquid_assets = mean_volume > MIN_MEAN_VOLUME  # Volume > 500,000
    liquid_dollar_assets = mean_dollar_volume > MIN_MEAN_DOLLAR_VOLUME  # $10M

    # Cumulative return (1 year, adjusted for 252 working days)
    for ticker in data.columns.levels[0]:
        # PCT change is scale independent :D
        data[(ticker, "Daily_Return")] = data[(ticker, "Close")].pct_change(fill_method=None)

    annual_return = data.xs("Daily_Return", level="Price", axis=1).mean() * 252

    # Historical volatility (annualized standard deviation)
    annual_volatility = data.xs("Daily_Return", level="Price", axis=1).std() * (252**0.5)

    # Filter by return and volatility
    performing_assets = annual_return > MIN_ANNUAL_RETURN  # Return > 5%
    stable_assets = (annual_volatility > MIN_ANNUAL_VOLATILITY) & (
        annual_volatility < MAX_ANNUAL_VOLATILITY
    )  # Vol > 10%, < 50%

    filtered = liquid_assets & liquid_dollar_assets & performing_assets & stable_assets

    categories_to_drop = data.columns.levels[0][~filtered]
    data_filtered = data.drop(columns=categories_to_drop, level=0)

    # Update dataframe columns
    data_filtered.columns = data_filtered.columns.remove_unused_levels()

    data_filtered.to_pickle(output_path)


@app.command()
def add_features(
    input_path: Path = RAW_DATA_DIR / "raw_filtered.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "extra_features.pkl",
):
    data_all: pd.DataFrame = pd.read_pickle(input_path)

    horizons = [2, 5, 60, 250, 1000]

    for ticker in data_all.columns.levels[0]:
        # Add target column
        data_all[(ticker, "Target")] = data_all[(ticker, "Close")].pct_change(fill_method=None)

        data = data_all.xs(ticker, level="Ticker", axis=1)

        # Add horizon features
        for horizon in horizons:
            rolling_avg = data.rolling(horizon).mean()

            ratio_column = f"Close_Ratio_{horizon}"
            data_all[(ticker, ratio_column)] = data["Close"] / rolling_avg["Close"]

            trend_column = f"Trend_{horizon}"
            data_all[(ticker, trend_column)] = data.shift(1).rolling(horizon).sum()["Target"]

    data_all.to_pickle(output_path)


if __name__ == "__main__":
    app()
