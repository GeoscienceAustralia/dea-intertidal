import logging
import yaml
import fsspec
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd
from pathlib import Path


def configure_logging(name: str = "DEA Intertidal") -> logging.Logger:
    """
    Configure logging for the application.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file and returns data as a nested dictionary.

    config_path can be a path or URL to a web accessible YAML file
    """
    with fsspec.open(config_path, mode="r") as f:
        config = yaml.safe_load(f)
    return config


def round_date_strings(date, round_type="end"):
    """
    Round a date string up or down to the start or end of a given time
    period.

    Parameters
    ----------
    date : str
        Date string of variable precision (e.g. "2020", "2020-01",
        "2020-01-01").
    round_type : str, optional
        Type of rounding to perform. Valid options are "start" or "end".
        If "start", date is rounded down to the start of the time period.
        If "end", date is rounded up to the end of the time period.
        Default is "end".

    Returns
    -------
    date_rounded : str
        The rounded date string.

    Examples
    --------
    >>> round_date_strings('2020')
    '2020-12-31 00:00:00'

    >>> round_date_strings('2020-01', round_type='start')
    '2020-01-01 00:00:00'

    >>> round_date_strings('2020-01', round_type='end')
    '2020-01-31 00:00:00'
    """

    # Determine precision of input date string
    date_segments = len(date.split("-"))

    # If provided date has no "-", treat it as having year precision
    if date_segments == 1 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + YearBegin(0))
    elif date_segments == 1 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + YearEnd(0))

    # If provided date has one "-", treat it as having month precision
    elif date_segments == 2 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + MonthBegin(0))
    elif date_segments == 2 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + MonthEnd(0))

    # If more than one "-", then return date as-is
    elif date_segments > 2:
        date_rounded = date

    return date_rounded
