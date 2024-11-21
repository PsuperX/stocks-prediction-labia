from pathlib import Path
import yfinance as yf

import typer
from loguru import logger

from .config import RAW_DATA_DIR, CONFIG

app = typer.Typer()


@app.command()
def download_dataset(
    output_path: Path = RAW_DATA_DIR / "raw.pkl",
):
    data = yf.download(
        " ".join(CONFIG["indices"]),
        start="2010-01-01",
        end="2024-01-31",
        group_by="ticker",
        auto_adjust=True,
        repair=True,
    )
    logger.success("Downloaded dataset.")
    data.to_pickle(output_path)
    logger.success("Saved dataset.")


if __name__ == "__main__":
    app()
