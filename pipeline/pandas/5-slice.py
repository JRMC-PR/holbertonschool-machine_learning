#!/usr/bin/env python3
"""This module slices a dataframe along the
columns High and Close"""
import pandas as pd


def slice(df):
    """This function slices a dataframe along the
    columns High and Close.

    Args:
        df: the dataframe to slice

    Returns:
        The sliced dataframe
    """
    # Return the columns 'High' and 'Close'
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::50]
