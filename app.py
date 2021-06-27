"""
TOPIX100 の各銘柄について以下の値を集計し、「寄付きが上向き/下向き」かどうかで銘柄の傾向に違いがあるか確認する
翌日の始値上向き(up)/下向き(down)の陽線(posi)/陰線(nega)の合計値, 平均値, 行数
Usage:
    $ conda activate stock
    $ streamlit run ./app.py
"""
import streamlit as st

import glob
import datetime
from pathlib import Path

# import traceback

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns

# 日本語対応
font = {"family": "Yu Gothic"}
matplotlib.rc("font", **font)
plt.rcParams["font.family"] = "Yu Gothic"
# sns.set(font='Yu Gothic')


stock_dir = "TOPIX100_data"
stock_name_csv = "stock_name_TOPIX100.csv"


def get_stock_df(stock_id):
    csv = stock_dir + "/" + str(stock_id) + ".csv"
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
    # df["翌日の始値-当日の終値"] = ( (df[f"翌日の始値"] - df[f"終値"]) / df[f"終値"] ) * 100
    # df["翌日の終値-翌日の始値"] = ( (df[f"翌日の終値"] - df[f"翌日の始値"]) / df[f"翌日の始値"] ) * 100

    return df


def up_down_summary(stock_id, start_date="2020-01-01", end_date="2020-12-31"):
    df = get_stock_df(stock_id=stock_id)

    df = df[(df["日付"] >= start_date) & (df["日付"] <= end_date)]
    df = df[df["翌日の終値-翌日の始値"].notnull()]  # 欠損ではない行のみ

    # 翌日の 陽線/陰線 の合計
    all_count = df.shape[0]
    all_sum = sum(df["翌日の終値-翌日の始値"])

    # 翌日の始値上向き + 翌日陽線のみ
    df_up_posi = df[(df["翌日の始値-当日の終値"] > 0.0) & (df["翌日の終値-翌日の始値"] > 0.0)]
    up_posi_count = df_up_posi.shape[0]
    up_posi_sum = sum(df_up_posi["翌日の終値-翌日の始値"])

    # 翌日の始値上向き + 翌日陰線のみ
    df_up_nega = df[(df["翌日の始値-当日の終値"] > 0.0) & (df["翌日の終値-翌日の始値"] < 0.0)]
    up_nega_count = df_up_nega.shape[0]
    up_nega_sum = sum(df_up_nega["翌日の終値-翌日の始値"])

    # 翌日の始値下向き + 翌日陽線のみ
    df_down_posi = df[(df["翌日の始値-当日の終値"] < 0.0) & (df["翌日の終値-翌日の始値"] > 0.0)]
    down_posi_count = df_down_posi.shape[0]
    down_posi_sum = sum(df_down_posi["翌日の終値-翌日の始値"])

    # 翌日の始値下向き + 翌日陰線のみ
    df_down_nega = df[(df["翌日の始値-当日の終値"] < 0.0) & (df["翌日の終値-翌日の始値"] < 0.0)]
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
    dict_summary["up_posi_mean"] = round(
        up_posi_sum / up_posi_count, 3
    )  # 翌日の 陽線/陰線 の平均

    # 翌日の始値上向き + 翌日陰線のみ
    dict_summary["up_nega_count"] = up_nega_count  # 行数
    dict_summary["up_nega_sum"] = up_nega_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["up_nega_mean"] = round(
        up_nega_sum / up_nega_count, 3
    )  # 翌日の 陽線/陰線 の平均

    # 翌日の始値下向き + 翌日陽線のみ
    dict_summary["down_posi_count"] = down_posi_count  # 行数
    dict_summary["down_posi_sum"] = down_posi_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["down_posi_mean"] = round(
        down_posi_sum / down_posi_count, 3
    )  # 翌日の 陽線/陰線 の平均

    # 翌日の始値下向き + 翌日陰線のみ
    dict_summary["down_nega_count"] = down_nega_count  # 行数
    dict_summary["down_nega_sum"] = down_nega_sum  # 翌日の 陽線/陰線 の合計
    dict_summary["down_nega_mean"] = round(
        down_nega_sum / down_nega_count, 3
    )  # 翌日の 陽線/陰線 の平均

    dict_summary["翌日の始値上向き_sum"] = up_sum  # 翌日の始値上向きの 陽線/陰線 の合計
    dict_summary["翌日の始値上向き_mean"] = round(
        up_sum / (up_posi_count + up_nega_count), 3
    )  # 翌日の始値上向きの 陽線/陰線 の平均

    dict_summary["翌日の始値下向き_sum"] = down_sum  # 翌日の始値下向きの 陽線/陰線 の合計
    dict_summary["翌日の始値下向き_mean"] = round(
        down_sum / (down_posi_count + down_nega_count), 3
    )  # 翌日の始値下向きの 陽線/陰線 の平均

    dict_summary["翌日陽線の割合"] = round(
        (up_posi_count + down_posi_count) / all_count * 100, 1
    )  # 翌日陽線の割合
    dict_summary["翌日陰線の割合"] = round(
        (up_nega_count + down_nega_count) / all_count * 100, 1
    )  # 翌日陰線の割合

    dict_summary["翌日の始値上向きの割合"] = round(
        (up_posi_count + up_nega_count) / all_count * 100, 1
    )
    dict_summary["翌日の始値下向きの割合"] = round(
        (down_posi_count + down_nega_count) / all_count * 100, 1
    )

    dict_summary["翌日の始値上向きかつ陽線の割合"] = round(up_posi_count / all_count * 100, 1)
    dict_summary["翌日の始値上向きかつ陰線の割合"] = round(up_nega_count / all_count * 100, 1)
    dict_summary["翌日の始値下向きかつ陰線の割合"] = round(down_nega_count / all_count * 100, 1)
    dict_summary["翌日の始値下向きかつ陽線の割合"] = round(down_posi_count / all_count * 100, 1)

    return dict_summary


def plot_sort_type_up_down(_df, sort_type, ascending=True):
    if sort_type == "sum":
        _sort_type = "合計"
    elif sort_type == "mean":
        _sort_type = "平均"
    else:
        _sort_type = sort_type

    _df = _df.rename(
        columns={
            f"翌日の始値上向き_{sort_type}": f"翌日の始値上向き_{_sort_type}",
            f"翌日の始値下向き_{sort_type}": f"翌日の始値下向き_{_sort_type}",
        }
    )
    _df = _df[[f"name", f"翌日の始値上向き_{_sort_type}", f"翌日の始値下向き_{_sort_type}"]].set_index(
        "name"
    )
    _df = _df.sort_values(by=f"翌日の始値上向き_{_sort_type}", ascending=ascending)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    _df.plot.barh(ax=ax)
    ax.set_title(f"期間中の陽線陰線の{_sort_type}")
    ax.set_xlabel("円")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    return fig


@st.cache(allow_output_mutation=True)
def get_df_summary(
    start_date, end_date, stock_dir=stock_dir, stock_name_csv=stock_name_csv
):
    stock_ids = [Path(csv).stem for csv in glob.glob(stock_dir + "/*csv")]
    df_summary = None
    for stock_id in stock_ids:
        dict_summary = up_down_summary(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        if df_summary is None:
            df_summary = pd.DataFrame.from_dict(dict_summary, orient="index").T
        else:
            df_summary = df_summary.append(dict_summary, ignore_index=True)

    # 見にくいので 行数の列 は削除
    df_summary = df_summary[
        [
            "stock_id",
            "開始日",
            "終了日",
            "取引日数",
            "開始日の始値",
            "終了日の始値",
            "始値の平均",
            "翌日の陽線/陰線の合計",
            "翌日の陽線/陰線の平均",
            # "up_posi_sum", "up_nega_sum", "down_posi_sum", "down_nega_sum",
            "翌日の始値上向き_sum",
            "翌日の始値下向き_sum",
            # "up_posi_mean", "up_nega_mean", "down_posi_mean", "down_nega_mean",
            "翌日の始値上向き_mean",
            "翌日の始値下向き_mean",
            # "翌日陽線の割合", "翌日陰線の割合",
            # "翌日の始値上向きの割合", "翌日の始値下向きの割合",
            "翌日の始値上向きかつ陽線の割合",
            "翌日の始値上向きかつ陰線の割合",
            "翌日の始値下向きかつ陰線の割合",
            "翌日の始値下向きかつ陽線の割合",
        ]
    ]

    # 銘柄名つける
    stock_name = pd.read_csv(stock_name_csv, encoding="SHIFT-JIS", dtype="str")
    df_summary = pd.merge(stock_name, df_summary, on="stock_id")

    return df_summary


def main():
    st.markdown("# TOPIX100の各銘柄について翌日の陽線陰線の値を集計")

    # サイドバー
    st_start_date = st.sidebar.date_input(
        "開始日", datetime.datetime.strptime("2020-1-1", "%Y-%m-%d").date()
    )
    st_end_date = st.sidebar.date_input(
        "終了日", datetime.datetime.strptime("2020-12-31", "%Y-%m-%d").date()
    )
    st_price_limit = st.sidebar.slider("集計する1銘柄の株価の上限", 0, 100000, (0, 5000))
    st_n_limit = st.sidebar.slider("表示する銘柄の件数", 1, 30, step=1, value=15)
    st_sort_type = st.sidebar.selectbox("可視化する価格の種類", ("sum", "mean"))

    try:
        # 集計したデータフレーム
        df_summary = get_df_summary(str(st_start_date), str(st_end_date))

        _df = df_summary[
            # (df_summary[f"翌日の始値上向き_{st_sort_type}"] > 0.0) &
            # (df_summary[f"翌日の始値下向き_{st_sort_type}"] > 0.0) &
            (df_summary[f"始値の平均"] >= st_price_limit[0])
            & (df_summary[f"始値の平均"] <= st_price_limit[1])  # 株価の高すぎる銘柄は除く
        ].sort_values(by=f"翌日の始値上向き_{st_sort_type}", ascending=False)

        _df = _df.head(st_n_limit)

        if st_sort_type == "sum":
            _str = "合計値"
        else:
            _str = "平均値"
        _str = f"翌日の始値上向きの{_str}が上位の銘柄"
        st.markdown("### " + _str)
        # plot
        st.pyplot(plot_sort_type_up_down(_df, st_sort_type))

        # table
        st.table(_df)
    except:
        st.markdown("## サイドバー値がおかしいためエラー")
        # traceback.print_exc()


if __name__ == "__main__":
    from matplotlib import font_manager

    for i in font_manager.fontManager.ttflist:
        if ".ttc" in i.fname:
            print(i)

    main()
