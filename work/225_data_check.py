# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

stock_dir = "../225_data"
stock_id = 1333
csv = stock_dir + "/" + str(stock_id) + ".csv"
df = pd.read_csv(
    csv,
    encoding="SHIFT-JIS",
    parse_dates=["日付"],
    na_values=["-"],
    dtype="float",
)
df
