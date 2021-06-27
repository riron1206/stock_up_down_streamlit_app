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

# # TOPIX100 の各銘柄について以下の値を集計し、「寄付きが上向き/下向き」かどうかで銘柄の傾向に違いがあるか確認する
# <br>
#
# ### 翌日の始値上向き(up)/下向き(down)の陽線(posi)/陰線(nega)の合計値, 平均値, 行数

# +
import os
import numpy as np
import pandas as pd

import glob
from pathlib import Path

from tqdm.auto import tqdm

# +
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語対応
plt.rcParams['font.family'] = 'Yu Gothic'
sns.set(font='Yu Gothic')

#pd.options.plotting.backend = "plotly"
# -

pd.set_option('display.max_columns', None)

data_dir = r"C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\指数乖離投資について\TOPIX100_10年分データ_edit"

stock_ids = [Path(csv).stem for csv in glob.glob(data_dir + "/*csv")]


# + code_folding=[0]
def get_stock_df(stock_id):
    csv = data_dir + "/" + str(stock_id) + ".csv"
    df = pd.read_csv(
        csv,
        encoding="SHIFT-JIS",
        sep="\t",
        parse_dates=["日付"],
        na_values=["-"],
        dtype="float",
    )
    df["銘柄"] = Path(csv).stem
    
    df = df.sort_values(by=["日付"])
    
    df[f"翌日の始値"] = df["始値"].shift(-1)
    df[f"翌日の終値"] = df["終値"].shift(-1)
    
    # 値の差をそのままつかう場合
    df["翌日の始値-当日の終値"] = df[f"翌日の始値"] - df[f"終値"]
    df["翌日の終値-翌日の始値"] = df[f"翌日の終値"] - df[f"翌日の始値"]
    # 値の差の割合%を取る場合
    #df["翌日の始値-当日の終値"] = ( (df[f"翌日の始値"] - df[f"終値"]) / df[f"終値"] ) * 100 
    #df["翌日の終値-翌日の始値"] = ( (df[f"翌日の終値"] - df[f"翌日の始値"]) / df[f"翌日の始値"] ) * 100 
    
    return df


# + code_folding=[0]
def up_down_summary(stock_id, start_date="2020-01-01", end_date="2020-12-31"):
    df = get_stock_df(stock_id=stock_id)
    
    df = df[(df["日付"] >= start_date) & (df["日付"] <= end_date)]
    df = df[df["翌日の終値-翌日の始値"].notnull()]  # 欠損ではない行のみ
    
    # 翌日の 陽線/陰線 の合計
    all_count = df.shape[0]
    all_sum = sum(df["翌日の終値-翌日の始値"])
    
    # 翌日の始値上向き + 翌日陽線のみ
    df_up_posi = df[
        (df["翌日の始値-当日の終値"] > 0.0) &
        (df["翌日の終値-翌日の始値"] > 0.0)
    ]
    up_posi_count = df_up_posi.shape[0]
    up_posi_sum = sum(df_up_posi["翌日の終値-翌日の始値"])
    
    # 翌日の始値上向き + 翌日陰線のみ
    df_up_nega = df[
        (df["翌日の始値-当日の終値"] > 0.0) &
        (df["翌日の終値-翌日の始値"] < 0.0)
    ]
    up_nega_count = df_up_nega.shape[0]
    up_nega_sum = sum(df_up_nega["翌日の終値-翌日の始値"])

    # 翌日の始値下向き + 翌日陽線のみ
    df_down_posi = df[
        (df["翌日の始値-当日の終値"] < 0.0) &
        (df["翌日の終値-翌日の始値"] > 0.0)
    ]
    down_posi_count = df_down_posi.shape[0]
    down_posi_sum = sum(df_down_posi["翌日の終値-翌日の始値"])

    # 翌日の始値下向き + 翌日陰線のみ
    df_down_nega = df[
        (df["翌日の始値-当日の終値"] < 0.0) &
        (df["翌日の終値-翌日の始値"] < 0.0)
    ]
    down_nega_count = df_down_nega.shape[0]
    down_nega_sum = sum(df_down_nega["翌日の終値-翌日の始値"])

    # 翌日の始値上向きの 陽線/陰線 の合計
    up_sum = up_posi_sum + up_nega_sum

    # 翌日の始値下向きの 陽線/陰線 の合計
    down_sum = down_posi_sum + down_nega_sum
    
    
    dict_summary = dict()
    dict_summary["stock_id"] = stock_id
    dict_summary["開始日"] = start_date
    dict_summary["終了日"] = end_date
    
    dict_summary["開始日の始値"] = int(df.iloc[0]["始値"])
    dict_summary["終了日の始値"] = int(df.iloc[-1]["始値"])
    dict_summary["始値の平均"] = int(df["始値"].mean())
    
    dict_summary["取引日数"] = all_count  # 行数
    dict_summary["翌日の陽線/陰線の合計"] = all_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["翌日の陽線/陰線の平均"] = round(all_sum / all_count, 3)  # 翌日の 陽線/陰線 の平均
    
    # 翌日の始値上向き + 翌日陽線のみ
    dict_summary["up_posi_count"] = up_posi_count  # 行数
    dict_summary["up_posi_sum"] = up_posi_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["up_posi_mean"] = round(up_posi_sum / up_posi_count, 3) # 翌日の 陽線/陰線 の平均
    
    # 翌日の始値上向き + 翌日陰線のみ
    dict_summary["up_nega_count"] = up_nega_count  # 行数
    dict_summary["up_nega_sum"] = up_nega_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["up_nega_mean"] = round(up_nega_sum / up_nega_count, 3)  # 翌日の 陽線/陰線 の平均
    
    # 翌日の始値下向き + 翌日陽線のみ
    dict_summary["down_posi_count"] = down_posi_count  # 行数
    dict_summary["down_posi_sum"] = down_posi_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["down_posi_mean"] = round(down_posi_sum / down_posi_count, 3)  # 翌日の 陽線/陰線 の平均
    
     # 翌日の始値下向き + 翌日陰線のみ 
    dict_summary["down_nega_count"] = down_nega_count  # 行数
    dict_summary["down_nega_sum"] = down_nega_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["down_nega_mean"] = round(down_nega_sum / down_nega_count, 3)  # 翌日の 陽線/陰線 の平均
    
    dict_summary["翌日の始値上向き_sum"] = up_sum  # 翌日の始値上向きの 陽線/陰線 の合計
    dict_summary["翌日の始値上向き_mean"] = round(up_sum / (up_posi_count + up_nega_count), 3)  # 翌日の始値上向きの 陽線/陰線 の平均
    
    dict_summary["翌日の始値下向き_sum"] = down_sum  # 翌日の始値下向きの 陽線/陰線 の合計
    dict_summary["翌日の始値下向き_mean"] = round(down_sum / (down_posi_count + down_nega_count), 3)  # 翌日の始値下向きの 陽線/陰線 の平均
    
    dict_summary["翌日陽線の割合"] = round((up_posi_count + down_posi_count) / all_count * 100, 1)  # 翌日陽線の割合
    dict_summary["翌日陰線の割合"] = round((up_nega_count + down_nega_count) / all_count * 100, 1)  # 翌日陰線の割合

    dict_summary["翌日の始値上向きの割合"] = round((up_posi_count + up_nega_count) / all_count * 100, 1)
    dict_summary["翌日の始値下向きの割合"] = round((down_posi_count + down_nega_count) / all_count * 100, 1)
    
    dict_summary["翌日の始値上向きかつ陽線の割合"] = round(up_posi_count / all_count * 100, 1) 
    dict_summary["翌日の始値上向きかつ陰線の割合"] = round(up_nega_count / all_count * 100, 1) 
    dict_summary["翌日の始値下向きかつ陰線の割合"] = round(down_nega_count / all_count * 100, 1)
    dict_summary["翌日の始値下向きかつ陽線の割合"] = round(down_posi_count / all_count * 100, 1)
    
    return dict_summary
# -

# debug
df = get_stock_df(stock_id=1925)
display(df)
dict_summary = up_down_summary(stock_id=1925, start_date="2020-01-01", end_date="2020-12-31")
display(dict_summary)

# +
df_summary = None
for stock_id in tqdm(stock_ids):
    dict_summary = up_down_summary(stock_id=stock_id, start_date="2020-01-01", end_date="2020-12-31")
    if df_summary is None:
        df_summary = pd.DataFrame.from_dict(dict_summary, orient='index').T
    else:
        df_summary = df_summary.append(dict_summary, ignore_index=True)

df_summary.head(5)
# -

# 見にくいので 行数の列 は削除
df_summary = df_summary[["stock_id", "開始日", "終了日", 
                         "取引日数",
                         "開始日の始値", "終了日の始値", "始値の平均",
                         "翌日の陽線/陰線の合計", 
                         "翌日の陽線/陰線の平均",
                         #"up_posi_sum", "up_nega_sum", "down_posi_sum", "down_nega_sum", 
                         "翌日の始値上向き_sum", "翌日の始値下向き_sum", 
                         #"up_posi_mean", "up_nega_mean", "down_posi_mean", "down_nega_mean", 
                         "翌日の始値上向き_mean", "翌日の始値下向き_mean", 
                         #"翌日陽線の割合", "翌日陰線の割合",
                         #"翌日の始値上向きの割合", "翌日の始値下向きの割合",
                         "翌日の始値上向きかつ陽線の割合", "翌日の始値上向きかつ陰線の割合", "翌日の始値下向きかつ陰線の割合", "翌日の始値下向きかつ陽線の割合",
                        ]]

# 銘柄名つける
stock_name = pd.read_csv("../stock_name_TOPIX100.csv", encoding="SHIFT-JIS", dtype="str")
df_summary = pd.merge(stock_name, df_summary, on="stock_id")



x = df_summary.iloc[0]
u_p = x["翌日の始値上向きかつ陽線の割合"]
u_n = x["翌日の始値上向きかつ陰線の割合"]
d_n = x["翌日の始値下向きかつ陰線の割合"]
d_p = x["翌日の始値下向きかつ陽線の割合"]
plt.pie([u_p, u_n, d_n, d_p], 
        labels=["翌日の始値上向きかつ陽線の割合", "翌日の始値上向きかつ陰線の割合", "翌日の始値下向きかつ陰線の割合", "翌日の始値下向きかつ陽線の割合"], 
        counterclock=False, 
        startangle=90, 
        autopct="%1.1f%%")
plt.show()

# +
#plt.barh(["翌日の始値上向きかつ陽線の割合", "翌日の始値上向きかつ陰線の割合", "翌日の始値下向きかつ陰線の割合", "翌日の始値下向きかつ陽線の割合"], 
#         [u_p, u_n, d_n, d_p], 
#         align='center')

_df = pd.DataFrame({"翌日の始値上向きかつ陽線の割合": [u_p], 
                    "翌日の始値上向きかつ陰線の割合": [u_n], 
                    "翌日の始値下向きかつ陰線の割合": [d_n], 
                    "翌日の始値下向きかつ陽線の割合": [d_p]})
_df = _df.T.sort_values(by=0)
display(_df)
_df.plot.barh(legend=False)
#_df.plot.barh()

plt.title(f'{x["name"]} {x["stock_id"]}')
plt.xlabel('%')
plt.show()
# -



def plot_sort_type_up_down(_df, sort_type, ascending=True):
    _df = _df.head(10)[[f"name", f"翌日の始値上向き_{sort_type}", f"翌日の始値下向き_{sort_type}"]].set_index('name')
    _df = _df.sort_values(by=f"翌日の始値上向き_{sort_type}", ascending=ascending)
    
    _df.plot.barh(figsize=(8, 10))
    plt.xlabel(f'円（期間中の陽線陰線の{sort_type}）')
    plt.show()


# ### df_summary の各列の意味
# - start_date: 開始日
# - end_date: 終了日
# - all_sum: 期間内での全陽線/陰線の合計値
# - all_mean: 期間内での全陽線/陰線の平均値。値が1以下の場合は単位%
# - up_sum: 期間内での翌日の寄付き上向きの陽線/陰線の合計値
# - up_mean: 期間内での翌日の寄付き上向きの陽線/陰線の平均値。値が1以下の場合は単位%
# - down_sum: 期間内での翌日の寄付き下向きの陽線/陰線の合計値
# - down_mean: 期間内での翌日の寄付き下向きの陽線/陰線の平均値。値が1以下の場合は単位%
# - posi_count_percent: 期間内での翌日陽線の割合(%)
# - nega_count_percent: 期間内での翌日陰線の割合(%)

sort_type = "sum"
#sort_type = "mean"
price_limit = 5000.0

# 翌日の始値上向きの 陽線/陰線 の合計値大きいのだけ
_df = df_summary.sort_values(by=f"翌日の始値上向き_{sort_type}", ascending=False).head(15)
display(_df)

plot_sort_type_up_down(_df, sort_type)

# 翌日の始値上向きの 陽線/陰線 > 0.0 かつ 翌日の始値下向きの 陽線/陰線 < 0.0 のだけ
_df = df_summary[
    (df_summary[f"翌日の始値上向き_{sort_type}"] > 0.0) &
    (df_summary[f"翌日の始値下向き_{sort_type}"] < 0.0) &
    (df_summary[f"始値の平均"] < price_limit)  # 株価の高すぎる銘柄は除く
].sort_values(by=f"翌日の始値上向き_{sort_type}", ascending=False).head(15)
display(_df)

plot_sort_type_up_down(_df, sort_type)

# 翌日で買いでデイトレしとけばほぼ勝てた銘柄
_df = df_summary[
    (df_summary[f"翌日の始値上向き_{sort_type}"] > 0.0) &
    (df_summary[f"翌日の始値下向き_{sort_type}"] > 0.0) &
    (df_summary[f"始値の平均"] < price_limit)  # 株価の高すぎる銘柄は除く
].sort_values(by=f"翌日の始値上向き_{sort_type}", ascending=False)
display(_df)

plot_sort_type_up_down(_df, sort_type)

# 翌日売りでデイトレしとけばほぼ勝てた銘柄
_df = df_summary[
    (df_summary[f"翌日の始値上向き_{sort_type}"] < 0.0) &
    (df_summary[f"翌日の始値下向き_{sort_type}"] < 0.0) &
    (df_summary[f"始値の平均"] < price_limit)  # 株価の高すぎる銘柄は除く
].sort_values(by=f"翌日の始値下向き_{sort_type}", ascending=True)
display(_df)

plot_sort_type_up_down(_df, sort_type, ascending=False)


