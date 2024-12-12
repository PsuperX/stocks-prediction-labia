from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.decomposition import PCA
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

# fmt: off
STOCK_SECTOR = {
    'Industrials': ['MMM',   'AOS',   'ALLE',   'AMTM',   'AME',   'ADP',   'AXON',   'BA',   'BR',   'BLDR',   'CHRW',   'CARR',   'CAT',   'CTAS',   'CPRT',   'CSX',   'CMI',   'DAY',   'DE',   'DAL',   'DOV',   'ETN',   'EMR',   'EFX',   'EXPD',   'FAST',   'FDX',   'FTV',   'GE',   'GEV',   'GNRC',   'GD',   'HON',   'HWM',   'HUBB',   'HII',   'IEX',   'ITW',   'IR',   'JBHT',   'J',   'JCI',   'LHX',   'LDOS',   'LMT',   'MAS',   'NDSN',   'NSC',   'NOC',   'ODFL',   'OTIS',   'PCAR',   'PH',   'PAYX',   'PAYC',   'PNR',   'PWR',   'RTX',   'RSG',   'ROK',   'ROL',   'SNA',   'LUV',   'SWK',   'TXT',   'TT',   'TDG',   'UBER',   'UNP',   'UAL',   'UPS',   'URI',   'VLTO',   'VRSK',   'GWW',   'WAB',   'WM',   'XYL'],
    'Health Care': ['ABT',   'ABBV',   'A',   'ALGN',   'AMGN',   'BAX',   'BDX',   'TECH',   'BIIB',   'BSX',   'BMY',   'CAH',   'CTLT',   'COR',   'CNC',   'CRL',   'CI',   'COO',   'CVS',   'DHR',   'DVA',   'DXCM',   'EW',   'ELV',   'GEHC',   'GILD',   'HCA',   'HSIC',   'HOLX',   'HUM',   'IDXX',   'INCY',   'PODD',   'ISRG',   'IQV',   'JNJ',   'LH',   'LLY',   'MCK',   'MDT',   'MRK',   'MTD',   'MRNA',   'MOH',   'PFE',   'DGX',   'REGN',   'RMD',   'RVTY',   'SOLV',   'STE',   'SYK',   'TFX',   'TMO',   'UNH',   'UHS',   'VRTX',   'VTRS',   'WAT',   'WST',   'ZBH',   'ZTS'],
    'Information Technology': ['ACN',   'ADBE',   'AMD',   'AKAM',   'APH',   'ADI',   'ANSS',   'AAPL',   'AMAT',   'ANET',   'ADSK',   'AVGO',   'CDNS',   'CDW',   'CSCO',   'CTSH',   'GLW',   'CRWD',   'DELL',   'ENPH',   'EPAM',   'FFIV',   'FICO',   'FSLR',   'FTNT',   'IT',   'GEN',   'GDDY',   'HPE',   'HPQ',   'IBM',   'INTC',   'INTU',   'JBL',   'JNPR',   'KEYS',   'KLAC',   'LRCX',   'MCHP',   'MU',   'MSFT',   'MPWR',   'MSI',   'NTAP',   'NVDA',   'NXPI',   'ON',   'ORCL',   'PLTR',   'PANW',   'PTC',   'QRVO',   'QCOM',   'ROP',   'CRM',   'STX',   'NOW',   'SWKS',   'SMCI',   'SNPS',   'TEL',   'TDY',   'TER',   'TXN',   'TRMB',   'TYL',   'VRSN',   'WDC',   'ZBRA'],
    'Utilities': ['AES',   'LNT',   'AEE',   'AEP',   'AWK',   'ATO',   'CNP',   'CMS',   'ED',   'CEG',   'D',   'DTE',   'DUK',   'EIX',   'ETR',   'EVRG',   'ES',   'EXC',   'FE',   'NEE',   'NI',   'NRG',   'PCG',   'PNW',   'PPL',   'PEG',   'SRE',   'SO',   'VST',   'WEC',   'XEL'],
    'Financials': ['AFL',   'ALL',   'AXP',   'AIG',   'AMP',   'AON',   'ACGL',   'AJG',   'AIZ',   'BAC',   'BRK.B',   'BLK',   'BX',   'BK',   'BRO',   'COF',   'CBOE',   'SCHW',   'CB',   'CINF',   'C',   'CFG',   'CME',   'CPAY',   'DFS',   'ERIE',   'EG',   'FDS',   'FIS',   'FITB',   'FI',   'BEN',   'GPN',   'GL',   'GS',   'HIG',   'HBAN',   'ICE',   'IVZ',   'JKHY',   'JPM',   'KEY',   'KKR',   'L',   'MTB',   'MKTX',   'MMC',   'MA',   'MET',   'MCO',   'MS',   'MSCI',   'NDAQ',   'NTRS',   'PYPL',   'PNC',   'PFG',   'PGR',   'PRU',   'RJF',   'RF',   'SPGI',   'STT',   'SYF',   'TROW',   'TRV',   'TFC',   'USB',   'V',   'WRB',   'WFC',   'WTW'],
    'Materials': ['APD',   'ALB',   'AMCR',   'AVY',   'BALL',   'CE',   'CF',   'CTVA',   'DOW',   'DD',   'EMN',   'ECL',   'FMC',   'FCX',   'IFF',   'IP',   'LIN',   'LYB',   'MLM',   'MOS',   'NEM',   'NUE',   'PKG',   'PPG',   'SHW',   'SW',   'STLD',   'VMC'],
    'Consumer Discretionary': ['ABNB',   'AMZN',   'APTV',   'AZO',   'BBY',   'BKNG',   'BWA',   'CZR',   'KMX',   'CCL',   'CMG',   'DRI',   'DECK',   'DPZ',   'DHI',   'EBAY',   'EXPE',   'F',   'GRMN',   'GM',   'GPC',   'HAS',   'HLT',   'HD',   'LVS',   'LEN',   'LKQ',   'LOW',   'LULU',   'MAR',   'MCD',   'MGM',   'MHK',   'NKE',   'NCLH',   'NVR',   'ORLY',   'POOL',   'PHM',   'RL',   'ROST',   'RCL',   'SBUX',   'TPR',   'TSLA',   'TJX',   'TSCO',   'ULTA',   'WYNN',   'YUM'],
    'Real Estate': ['ARE',   'AMT',   'AVB',   'BXP',   'CPT',   'CBRE',   'CSGP',   'CCI',   'DLR',   'EQIX',   'EQR',   'ESS',   'EXR',   'FRT',   'DOC',   'HST',   'INVH',   'IRM',   'KIM',   'MAA',   'PLD',   'PSA',   'O',   'REG',   'SBAC',   'SPG',   'UDR',   'VTR',   'VICI',   'WELL',   'WY'],
    'Communication Services': ['GOOGL',   'GOOG',   'T',   'CHTR',   'CMCSA',   'EA',   'FOXA',   'FOX',   'IPG',   'LYV',   'MTCH',   'META',   'NFLX',   'NWSA',   'NWS',   'OMC',   'PARA',   'TMUS',   'TTWO',   'VZ',   'DIS',   'WBD'],
    'Consumer Staples': ['MO',   'ADM',   'BF.B',   'BG',   'CPB',   'CHD',   'CLX',   'KO',   'CL',   'CAG',   'STZ',   'COST',   'DG',   'DLTR',   'EL',   'GIS',   'HSY',   'HRL',   'K',   'KVUE',   'KDP',   'KMB',   'KHC',   'KR',   'LW',   'MKC',   'TAP',   'MDLZ',   'MNST',   'PEP',   'PM',   'PG',   'SJM',   'SYY',   'TGT',   'TSN',   'WBA',   'WMT'],
    'Energy': ['APA',   'BKR',   'CVX',   'COP',   'CTRA',   'DVN',   'FANG',   'EOG',   'EQT',   'XOM',   'HAL',   'HES',   'KMI',   'MPC',   'OXY',   'OKE',   'PSX',   'SLB',   'TRGP',   'TPL',   'VLO',   'WMB']} 
# fmt: on

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
    delta = data.diff()

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
        new_features[(ticker, "RSI")] = calculate_rsi(data["Target"])

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


def extract_features_with_scale(data_all: pd.DataFrame) -> pd.DataFrame:
    data_all = data_all.copy(deep=False)

    # horizons = [2, 5, 60, 250, 1000]
    horizons = [2, 5, 60, 250]

    new_features = {}

    for ticker in data_all.columns.levels[0]:
        # Add target column
        data_all[(ticker, "Target")] = data_all[(ticker, "Close")].diff()

        data = data_all.xs(ticker, level="Ticker", axis=1)

        # Add horizon features
        for horizon in horizons:
            rolling_avg = data["Close"].rolling(horizon).mean()

            # How much the close price has changed compared to previous days
            ratio_column = f"Close_Ratio_{horizon}"
            new_features[(ticker, ratio_column)] = rolling_avg

            # How much the price went up in the previous days
            trend_column = f"Trend_{horizon}"
            new_features[(ticker, trend_column)] = data["Target"].shift(1).rolling(horizon).sum()

        new_features[(ticker, "EMA")] = data["Target"].ewm(span=12, adjust=False).mean()
        new_features[(ticker, "RSI")] = calculate_rsi(data["Target"])

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


def rfe_feature_selection(
    df: pd.DataFrame, min_features_to_select: int, step: int, verbose: bool = False
) -> Tuple[pd.DataFrame, Pipeline]:
    clf = Lasso(alpha=0.1)
    feature_selector = RFE(
        estimator=clf,
        step=step,
        n_features_to_select=min_features_to_select,
        verbose=verbose,
    )

    return feature_selection(df, feature_selector, verbose)


def rfecv_feature_selection(
    df: pd.DataFrame,
    clf: BaseEstimator,
    min_features_to_select: int,
    step: int,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Pipeline]:
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

    return feature_selection(df, feature_selector, verbose)


def pca_feature_selection(
    df: pd.DataFrame,
    n_components: int,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Pipeline]:
    pca = PCA(n_components=n_components, random_state=42)
    pipeline = Pipeline(
        [
            ("normalize", RobustScaler()),
            ("feature_selection", pca),
        ],
        verbose=verbose,
    )

    y = df.xs("Target", level="Price", axis=1).shift(-1).dropna()  # Predicting next day's price
    X = df.iloc[:-1]  # Drop the last row of X to align with y

    new_df = pipeline.fit_transform(X, y)

    return new_df, pipeline


def feature_selection(
    df: pd.DataFrame,
    feature_selector: BaseEstimator,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Pipeline]:
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
