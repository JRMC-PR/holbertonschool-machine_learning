#!/usr/bin/env python3
"""This module concatenates two dataframes"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """This function concatenates two dataframes

    Args:
        df: the dataframe to concatenate

    Returns:
        The concatenated dataframe
    """
    # Remove the "Weighted_Price" column
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # Fill missing values in the "Close" column with the previous row's value
    if "Close" in df.columns:
        df["Close"] = df["Close"].ffill()

    # Fill missing values in these columns with corresponding "Close" values
    for column in ["High", "Low", "Open"]:
        if column in df.columns:
            df[column] = df[column].fillna(df["Close"])

    # Set missing values in the Volume columns to 0
    for column in ["Volume_(BTC)", "Volume_(Currency)"]:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df
