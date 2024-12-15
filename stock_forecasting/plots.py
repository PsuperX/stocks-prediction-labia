from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model

import typer
from loguru import logger
from tqdm import tqdm

from .config import FIGURES_DIR, PROCESSED_DATA_DIR, CONFIG
from .utils import waveletSmooth, wavelet_transform

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


def plot_multi_task_history(history):
    # Extract data from history
    history_dict = history.history

    # Get the train/val loss
    train_loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])

    # Get the MSE for all tasks, assuming 'mse_task1', 'mse_task2', etc., are in the history
    task_mse = [key for key in history_dict.keys() if "mse" in key and "val" in key]
    task_mse_values = {task: history_dict[task] for task in task_mse}

    # Get the R² score for all tasks, assuming 'r2_task1', 'r2_task2', etc., are in the history
    task_r2 = [key for key in history_dict.keys() if "R2" in key and "val" in key]
    task_r2_values = {task: history_dict[task] for task in task_r2}

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot train vs validation loss
    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(val_loss, label="Validation Loss", linestyle="--")
    axs[0].set_title("Train vs Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot MSE for all tasks
    for task, mse_values in task_mse_values.items():
        axs[1].plot(mse_values, label=task)
    axs[1].set_title("Mean Squared Error (MSE) for All Tasks")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("MSE")
    # axs[1].legend()

    # Plot R² score for all tasks
    for task, r2_values in task_r2_values.items():
        axs[2].plot(r2_values, label=task)
    axs[2].set_title("R² Score for All Tasks")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("R² Score")
    # axs[2].legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def plot_wavelet(df: pd.DataFrame):
    train = df.xs("Close", level="Price", axis=1).values[:100, :1]
    test = df.xs("Close", level="Price", axis=1).values[100:150, :1]

    denoised_train, denoised_test = wavelet_transform(train, test)
    plt.figure(figsize=(14, 7))
    plt.plot(df.xs("Close", level="Price", axis=1).iloc[:150, :1].values, label="Original")
    plt.plot(np.arange(100), denoised_train, label="Denoised train")
    plt.plot(np.arange(100, 150), denoised_test, label="Denoised test")
    plt.legend()
    plt.title("Wavelet transform")


def plot_model_weights(model: Model):
    plt.figure(figsize=(10, 6))

    # Loop through each layer and extract its weights
    for layer in model.layers[100:110]:
        # Get the weights of the layer (this will return a list of arrays for each layer)
        weights = layer.get_weights()

        # If weights exist for this layer, plot their histogram
        if len(weights) > 0:
            for w in weights:
                w_flattened = w.flatten()
                plt.hist(w_flattened, bins=50, alpha=0.5, label=f"{layer.name} weights")

    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Weights in the Model")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    app()
