from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import typer
from loguru import logger
from tqdm import tqdm

from .config import FIGURES_DIR, PROCESSED_DATA_DIR, CONFIG

app = typer.Typer()


def plot_stock_stats(stats: pd.DataFrame):
    conf = CONFIG["stock_filter"]

    # Remove S&P 500 itself
    stats = stats.drop("^GSPC")

    # Plot
    ax = stats.plot(
        kind="bar",
        subplots=True,
        layout=(2, 2),
        title="Stock Stats",
        figsize=(18, 8),
        logy=True,
        legend=False,
        xticks=[],
    )

    # Add horizontal lines with the limits
    ax[0][0].axhline(y=conf["min_mean_volume"], color="red", linestyle=":", linewidth=2)
    ax[0][1].axhline(y=conf["min_mean_dollar_volume"], color="red", linestyle=":", linewidth=2)
    ax[1][0].axhline(y=conf["min_annual_return"], color="red", linestyle=":", linewidth=2)
    ax[1][1].axhline(y=conf["min_annual_volatility"], color="red", linestyle=":", linewidth=2)
    ax[1][1].axhline(y=conf["max_annual_volatility"], color="red", linestyle=":", linewidth=2)


def pairplot(df: pd.DataFrame):
    # Add a 'color' column based on the normalized target
    df = df.copy(deep=False)
    df.loc[:, "color"] = df["Next_Target"]

    # Pairplot with custom coloring
    pairplot = sns.pairplot(
        df,
        hue="color",
        plot_kws={"alpha": 0.8, "s": 50, "edgecolor": "k", "linewidth": 0.5},  # Point properties
        diag_kws={"hue": None, "color": ".3"},
        diag_kind="hist",
        palette=None,
    )

    pairplot.add_legend()

    # Show the plot
    plt.suptitle("Pairplot of features", y=1.02)


def plot_feature_ranking(df: pd.DataFrame, pipeline: Pipeline):
    ranking = pipeline["feature_selection"].ranking_

    df = pd.Series(ranking, index=df.columns)
    df = df.xs("Target", level="Price", axis=0)
    df.plot(kind="bar", title="Ranking of features with RFE\n(Lasso)", figsize=(17, 10), xticks=[])


def plot_rfecv_results(pipeline: Pipeline):
    cv_results = pd.DataFrame(pipeline["feature_selection"].cv_results_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean r2 score")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()


def plot_loss(history: dict, title: str = "Losses"):
    # Test 1 results
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Test loss")
    plt.title(title)
    plt.legend()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
