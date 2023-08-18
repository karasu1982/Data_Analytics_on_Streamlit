import streamlit as st
from dotenv import load_dotenv
import os
import duckdb

# 画像関連
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import japanize_matplotlib

# 分析関連
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# DuckDB設定
# _PROJECT_PATH="/".join( os.path.abspath(__file__).split("/")[0:-2] )
# load_dotenv(os.path.join(_PROJECT_PATH,".env"))

# con = duckdb.connect(database=os.path.join(_PROJECT_PATH,'db',os.environ['DB_NAME']), read_only=True)
# con= duckdb.connect(database="/content/colab_streamlit_csv_upload/db/database.db", read_only=False)

# ページタイトルと表示設定
st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis App")
st.write("#")

# データを読み込む
# st.subheader("データをアップロードする")
uploaded_file="tmp"

# サイドバー設定
if uploaded_file is not None:

    # DuckDB内のタイタニックデータを使う場合
    #df = con.execute("SELECT * FROM test2").df()

    # IRISデータを使う場合
    df=sns.load_dataset('iris')

    # ローカルのCSVを使う場合
    #df = pd.read_csv("/content/file.csv")

    st.sidebar.write("# ラベル")
    cols_list = list(df.columns)

    label = st.sidebar.selectbox(
    '予測する対象',
    cols_list)

    st.sidebar.write("# 特徴量")

    features = st.sidebar.multiselect(
    '予測に使う説明変数',
    cols_list,
    )

    # ラベル + 特徴量
    target_cols = list(set([label]+features))

    # そのうち数値項目の件数
    number_cols = list(df[target_cols].select_dtypes(include=np.number).columns)

    # ラベル
    st.sidebar.write("# ラベルの分布")

    ## グラフ描画エリアの設定
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[label]
    )

    fig.add_trace(
        go.Histogram(x=df[label], marker=dict(color='#7CFC00'), name=label),
        row=1, col=1
    )

    fig.update_layout(bargap=0.2)
    st.sidebar.plotly_chart(fig, use_container_width=True)

    #st.sidebar.write("##")
    #st.sidebar.write("### 欠損値")
    #st.sidebar.write("各カラムの欠損値")

    # 欠損値の表示
    #null_df = pd.DataFrame(df.isnull().sum(), columns=["null"])
    #st.sidebar.dataframe(null_df)

if uploaded_file is not None and len(features) > 0:

# 1.要約統計量の確認
    st.write("#")
    st.subheader("1. 要約統計量の確認")

    # ページカラムを2つに分割
    left_column, right_column = st.columns(2)

    # データサンプルの表示
    left_column.write("###")
    left_column.write("##### サンプルデータ")

    left_column.dataframe(df.head(5))

    # 要約統計量の表示
    right_column.write("###")
    right_column.write("##### 要約統計量 (数値データのみ)")

    if len(number_cols) > 0:
      right_column.dataframe(df[target_cols].describe())

# 2. 各データの分布/割合を確認
    st.write("#")
    st.subheader("2. 特徴量の分布")

    # ページカラムを2つに分割
    left_column, right_column = st.columns(2)

    ## 1行あたり何個のグラフを表示するか
    GRAPH_COLS = 1

    rows = int(math.ceil(len(features) / GRAPH_COLS))

    # グラフ描画エリアの設定
    fig = make_subplots(
        rows=rows, cols=GRAPH_COLS,
        subplot_titles=features
    )

    ## n行5列でグラフを描画
    for n, option in enumerate(features):

      fig = px.histogram(df, x=option, color=label , barmode="stack")

      #row = int(n // GRAPH_COLS) + 1
      #col = int(n % GRAPH_COLS)  + 1

      #fig.add_trace(
      #  px.histogram(df, x=option, color="Survived" , barmode="stack"),
      #  row=row, col=col
      #)

      #fig.add_trace(
      #    go.Histogram(x=df[option], name=option),
      #    row=row, col=col
      #)

      ## グラフエリアの縦横長とgapの設定
      fig.update_layout(bargap=0.2)

      if n %2 == 0:
        left_column.plotly_chart(fig, use_container_width=True)
      else:
        right_column.plotly_chart(fig, use_container_width=True)

# 3.2変数同士の関係
    st.write("#")
    st.subheader("3. 2変数同士の関係")

    # ページカラムを2つに分割
    left_column, right_column = st.columns(2)

    left_column.write("##### 相関係数")

    if len(number_cols) > 0:
      fig = px.imshow(df[number_cols].corr(), text_auto=True)
      left_column.plotly_chart(fig, use_container_width=True)

    #right_column.write("##### 散布図行列")
    #fig = px.scatter_matrix(df, dimensions = target_cols)
    #right_column.plotly_chart(fig, use_container_width=True)

    right_column.write("##### ペアプロット図 (数値データのみ)")

    fig = sns.pairplot(df[target_cols], hue=label)
    right_column.pyplot(fig)


# 4.予測モデルの構築
    st.write("#")
    st.subheader("4. 予測モデルの構築")

    if st.button("重回帰分析 開始"):
        # left_column.write("分析を開始しました。")
        x = df[features]
        y = df[label]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        multi_OLS = sm.OLS(y, sm.add_constant(x_scaled))
        result = multi_OLS.fit()

        # ページカラムを2つに分割
        left_column, right_column = st.columns(2)

        left_column.write("分析が終了しました。結果を表示します。")
        right_column.text(result.summary())

        left_column.write(f"自由度調整済決定係数は{result.rsquared_adj:.2f}でした。")

        pred = result.predict(sm.add_constant(x_scaled))
        num = list(range(0, len(x)))

        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=num, y=y,
                            mode='markers',
                            name='target'))
        fig.add_trace(go.Scatter(x=num, y=pred,
                            mode='markers',
                            name='prediction'))
        fig.update_xaxes(title_text="Sample No.")
        fig.update_yaxes(title_text="Target / Prediction Value")
        left_column.plotly_chart(fig, use_container_width=True)
