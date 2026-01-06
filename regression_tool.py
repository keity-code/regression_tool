import streamlit as st 
import pandas as pd   
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns  
import matplotlib
import matplotlib.pyplot as plt 
import platform
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import google.generativeai as genai


# Gemini API キーの設定
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except AttributeError:
    st.error("Gemini API キーが設定されていません。secret.tomlにGOOGLE_API_KEYを設定してください。")
    st.stop() # キーがない場合は処理を停止する


# モデルの初期化1.5-pro-001
model_AI = genai.GenerativeModel('gemini-2.0-flash') 

page = st.sidebar.selectbox("機能を選択",["基礎統計量","二変量間の相関関係","回帰分析","主成分分析","統計用語の解説"])
st.title("統計分析ツール")

# 日本語フォント設定（Windows/Mac/Linuxで自動選択）
import platform
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Meiryo')                     # Windows
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='Hiragino Maru Gothic Pro')   # Mac
else:
    matplotlib.rc('font', family='IPAPGothic')                 # Linuxなど

# ぺーぎごとに処理を切り替える
if page == "基礎統計量":
    st.header("基礎統計量の表示")
    uploaded_file = st.file_uploader("CSV または Excelファイルをアップロードしてください。", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("データプレビュー")
        st.dataframe(df)

        # 基礎統計量
        st.write("基礎統計量")
        st.dataframe(df.describe())   

        # グラフ選択
        st.subheader("グラフを表示")
        graph_type = st.selectbox("グラフの種類を選択", ["円グラフ", "箱ひげ図", "ヒストグラム", "棒グラフ"])

        # 対象列を選択
        selected_col = st.selectbox("列を選択", df.columns)

        # グラフ描画
        fig, ax = plt.subplots()
        if graph_type == "円グラフ":
            if df[selected_col].dtype == "object" or df[selected_col].dtype.name == "category":
                df[selected_col].value_counts().plot.pie(autopct="%1.1f%%", ax = ax)
                ax.set_ylabel("")
                ax.set_title(f"{selected_col}の円グラフ")
            else:
                st.warning("円グラフはカテゴリーデータのみ対応しています。")
        elif graph_type == "箱ひげ図":
            sns.boxplot(y=df[selected_col], ax=ax)
            ax.set_title(f"{selected_col}の箱ひげ図")
        elif graph_type == "ヒストグラム":
            df[selected_col].hist(ax=ax, bins=20)
            ax.set_title(f"{selected_col}のヒストグラム")
        elif graph_type == "棒グラフ":
            if df[selected_col].dtype == "object":
                df[selected_col].value_counts().plot.bar(ax=ax)
                ax.set_title(f"{selected_col}の棒グラフ")
            else:
                st.warning("棒グラフはカテゴリーデータのみ対応しています。")
        st.pyplot(fig)                                  

elif page == "二変量間の相関関係":
    st.header("二変量間の相関関係分析")
    st.markdown("相関係数、有意確率を分析します。")

    uploaded_file = st.file_uploader("CSV または Excel ファイルをアップロードしてください。", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("データプレビュー")
        st.dataframe(df.head())

        # 数値列のみ抽出
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()

        if len(numeric_cols) >= 2:
            # マルチセレクトで複数の変数を選べるように変更
            selected_cols = st.multiselect("分析したい変数を2つ以上選択してください", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

            if len(selected_cols) >= 2:
                if st.button("相関分析実行"):
                    st.subheader("1. 相関行列（係数と有意差）")
                    st.markdown("注: `* p < .05`, `** p < .01` (両側検定)")

                    # 結果を格納するデータフレームの準備
                    corr_matrix_display = pd.DataFrame(index=selected_cols, columns=selected_cols) # 表示用（星付き文字列）
                    corr_matrix_numeric = pd.DataFrame(index=selected_cols, columns=selected_cols) # ヒートマップ用（数値）
                    
                    # 詳細リスト用
                    details_list = []

                    # 総当たりで計算
                    for col1 in selected_cols:
                        for col2 in selected_cols:
                            if col1 == col2:
                                corr_matrix_display.loc[col1, col2] = "1.000"
                                corr_matrix_numeric.loc[col1, col2] = 1.0
                                continue
                            
                            # ペアワイズで欠損値除去（SPSSの「ペアごとに除外」と同じ挙動）
                            temp_df = df[[col1, col2]].dropna()
                            n = len(temp_df)
                            
                            if n < 2:
                                corr_matrix_display.loc[col1, col2] = "-"
                                corr_matrix_numeric.loc[col1, col2] = 0
                            else:
                                r, p = stats.pearsonr(temp_df[col1], temp_df[col2])
                                
                                # 星の付与
                                star = ""
                                if p < 0.01:
                                    star = "**"
                                elif p < 0.05:
                                    star = "*"
                                
                                # 表示用データフレームに格納
                                corr_matrix_display.loc[col1, col2] = f"{r:.3f}{star}"
                                # ヒートマップ用データフレームに格納
                                corr_matrix_numeric.loc[col1, col2] = r

                                # 詳細リストに追加（重複を避けるため上三角成分のみ、または全ペア）
                                # ここでは見やすさのため全ペアリストにはせず、マトリックスで確認できない詳細用として保存
                                details_list.append({
                                    "変数1": col1,
                                    "変数2": col2,
                                    "相関係数(r)": r,
                                    "有意確率(p)": p,
                                    "度数(N)": n,
                                    "有意差(５％)": "有意差あり" if p < 0.05 else "有意差なし"
                                })

                    # 1. 相関行列の表示
                    st.dataframe(corr_matrix_display)

                    # 2. ヒートマップの表示
                    st.subheader("2. ヒートマップ")
                    fig, ax = plt.subplots(figsize=(10, 8)) # サイズ調整
                    sns.heatmap(corr_matrix_numeric.astype(float), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                    ax.set_title("相関行列ヒートマップ")
                    st.pyplot(fig)

                    # 3. 詳細データ（SPSS風の詳細情報）
                    st.subheader("3. 詳細データリスト (r, p, N)")
                    st.markdown("各ペアの詳細な数値です。")
                    
                    details_df = pd.DataFrame(details_list)
                    
                    # p値を見やすくフォーマット
                    details_df["相関係数(r)"] = details_df["相関係数(r)"].apply(lambda x: f"{x:.3f}")
                    details_df["有意確率(p)"] = details_df["有意確率(p)"].apply(lambda x: f"{x:.3e}" if x < 0.001 else f"{x:.3f}")
                    
                    # ユーザーが見やすいように、重複ペア（A-BとB-A）を除去する処理を入れるか、
                    # あるいは全ての組み合わせを表示するか。今回はわかりやすく全組み合わせを表示します。
                    st.dataframe(details_df)
                    
                    # データのダウンロード機能
                    csv = details_df.to_csv(index=False).encode('utf-8_sig')
                    st.download_button(
                        label="詳細データをCSVでダウンロード",
                        data=csv,
                        file_name='correlation_details.csv',
                        mime='text/csv',
                    )

            else:
                st.info("変数を2つ以上選択して、分析実行ボタンを押してください。")
        else:
            st.warning("数値データが2列以上あるファイルをアップロードしてください。")

elif page == "回帰分析":
    st.header("回帰分析")

    # CSVファイルのアップロード
    uploaded_file = st.file_uploader("CSV または Excel ファイルをアップロードしてください。" , type=["csv", "xlsx", "xls"])

     # ファイルが保存されたらセッションステートに保存
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['df_regression'] = df

    # セッションステートがデータフレームにある場合のみ処理を進める
    if "df_regression" in st.session_state:  
        df = st.session_state["df_regression"]      

        st.write("データレビュー：")
        st.dataframe(df)

    # 数値列のみを対象に選択肢を絞る
        numeric_cols = df.select_dtypes(include=["float" , "int"]).columns.tolist()

        if len(numeric_cols) >= 2:
           y_col = st.selectbox("従属変数(Y)" , numeric_cols)
           x_col = st.multiselect("独立変数(X)" ,  [col for col in numeric_cols if col != y_col])
        
           show_only_significant = st.checkbox("p値が0.1未満の変数だけ表示", value=False)

           if x_col:
            if st.button("回帰分析実行", key="run_regression" ):
                # 1. 分析に使うデータだけを一時的に抜き出す
                temp_df = df[[y_col] + x_col].copy()
                
                # 2. 欠損値（NaN）や無限大が含まれる行を削除する
                temp_df = temp_df.dropna()
                
                # データがなくなってしまった場合のガード（念のため）
                if temp_df.empty:
                    st.error("エラー：欠損値を除去した結果、データがなくなりました。入力データを確認してください。")
                    st.stop()

                # 3. きれいになったデータで X と y を定義しなおす
                X = temp_df[x_col]
                y = temp_df[y_col]

                X_const = sm.add_constant(X)    # 切片

                is_binary = y.dropna().isin([0,1]).all() and y.nunique() == 2

                if is_binary:
                   # ロジスティック回帰(目的変数が0/1のとき)
                   st.subheader("ロジスティック回帰分析")
                   model = sm.Logit(y, X_const).fit(disp=False)
                else:
                   # 通常の線形回帰
                   st.subheader("線形回帰分析")
                   model = sm.OLS(y, X_const).fit()

                st.write("モデル要約")
                st.text(model.summary())

                # 線形回帰のときだけ決定係数を表示
                r_squared = None
                if not is_binary:
                    r_squared = model.rsquared_adj
                    st.write(f"調整済み決定係数(Adjusted R²)：{r_squared:.3f}")
                    
                #各独立変数(X)と従属変数(y)の相関係数を計算    
                corrs = X.apply(lambda x: x.corr(y))    

                # 結果をDatFrameに整理
                summary_df = pd.DataFrame({
                    "回帰係数":model.params,
                    "標準化係数": model.params.index.map(corrs),
                    "p値":model.pvalues,
                    "標準誤差":model.bse,
                })

                if show_only_significant:
                    summary_df = summary_df[summary_df["p値"] < 0.1]

                def highlight_pvalue(val):
                    color = "background-color: #a6f3a6" if val < 0.1 else "background-color: #f3a6a6"
                    return color
                
                st.write("回帰結果")
                st.dataframe(summary_df.style.applymap(highlight_pvalue, subset=["p値"]))

                # 回帰直線の可視化
                if not is_binary and len(x_col) == 1:
                    st.write("回帰直線の可視化")
                    fig, ax = plt.subplots()
                    sns.regplot(x=df[x_col[0]], y=y, ax=ax, line_kws={"color":"red"})
                    ax.set_xlabel(x_col[0])
                    ax.set_ylabel(y_col)
                    ax.set_title("回帰直線と散布図")
                    st.pyplot(fig)    

                # 残差のヒストグラム
                if not is_binary:
                    st.write("### 残差のヒストグラム")
                    residuals = model.resid
                    fig2, ax2 = plt.subplots()
                    sns.histplot(residuals, kde=True, ax=ax2)
                    ax2.set_title("残差の分布")
                    st.pyplot(fig2)

                # ロジスティック回帰の分類精度
                accuracy = None
                if is_binary:
                    y_pred = model.predict(X_const) > 0.5
                    accuracy = accuracy_score(y, y_pred)
                    st.write(f"分類制度（正解率）： {accuracy:.3f}")

                # 全ての分析結果をセッションステートに保管
                st.session_state['regression_model'] = model
                st.session_state['regression_summary_df'] = summary_df
                st.session_state['regression_is_binary'] = is_binary
                st.session_state['regression_y_col'] = y_col
                st.session_state['regression_x_col'] = x_col
                st.session_state['regression_r_squared'] = r_squared
                st.session_state['regression_accuracy'] = accuracy  
           else:
            st.info("独立変数を選択し、「回帰分析実行」ボタンを押してください")      

            # AIによる推論・解釈の追加
            # 分析結果がセッションステートに存在する場合のみ表示
           if "regression_model" in st.session_state:
               st.subheader("AIによる分析結果の解釈")
               if st.button("AIに解釈を依頼", key="ask_ai_interpretation"):
                   # セッションステートから分析結果を読み込む
                    model = st.session_state['regression_model']
                    summary_df = st.session_state['regression_summary_df']
                    is_binary = st.session_state['regression_is_binary']
                    y_col = st.session_state['regression_y_col']
                    x_col = st.session_state['regression_x_col']
                    r_squared = st.session_state['regression_r_squared']
                    accuracy = st.session_state['regression_accuracy'] 

                    y = df[y_col]   
                    is_binary = y.dropna().isin([0,1]).all() and y.nunique() == 2               
                
                    # 回帰分析のタイプを判断
                    analysis_type = "ロジスティック回帰分析" if is_binary else "線形回帰分析"

                    # 結果をテキスト形式で整形
                    interpretation_prompt = f"""
                    あなたは経験豊富なデータサイエンティストです。統計初学者でもわかりやすい言葉を使ってください。
                    以下の統計分析結果を簡潔に解釈してください。
                    特に統計的に有意な変数に焦点を当て、得られる仮説や解釈を出力してください。

                    ---
                    **分析タイプ:** {analysis_type}
                    **目的変数:** {y_col}
                    **説明変数:** {', '.join(x_col)}

                    **回帰結果概要:**
                    {model.summary().as_text()}

                    **回帰係数とp値:**
                    """
                    for index, row in summary_df.iterrows():
                        interpretation_prompt += f"""
                        - 変数 '{index}': 回帰係数={row['回帰係数']:.3f}, p値={row['p値']:.3f}"""

                    if not is_binary:
                        interpretation_prompt += f"""
                    **調整済み決定係数(Adjusted R²):** {r_squared:.3f}"""
                    else:
                        interpretation_prompt += f"""
                    **分類精度（正解率）:** {accuracy:.3f}"""

                    interpretation_prompt += """
                    ---

                    上記の情報に基づいて、解釈と示唆を提供してください。
                    """

                    with st.spinner('AIが分析結果を解釈中です...'):
                     try:
                        response = model_AI.generate_content(interpretation_prompt)
                        st.markdown(response.text)
                     except Exception as e:
                        st.error(f"AIによる解釈中にエラーが発生しました: {e}")    


elif  page == "主成分分析(PCA)":
       st.header("主成分分析ページ")
       uploaded_file = st.file_uploader("CSV または Excel ファイルをアップロードしてください。", type=["csv", "xlsx"])
      
       if uploaded_file is not None:
       # ファイル読み込み
        if uploaded_file.name.endswith(".csv"):
         df = pd.read_csv(uploaded_file)
        else:
         df = pd.read_excel(uploaded_file)
      
        st.write("データプレビュー")
        st.dataframe(df)

        # 数値列のみ抽出
        numeric_cols = df.select_dtypes(include=["Float64", "int64"]).columns
        st.write("数値データのみをPCAに使用", list(numeric_cols))

        # 標準化
        X = df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # 分散説明率
        explained_var = pca.explained_variance_ratio_
        st.write("分散説明率：")
        for i, ratio in enumerate(explained_var):
           st.write(f"主成分{i+1}：{ratio:.3f}")

        # 累積寄与率プロット
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(explained_var) + 1), explained_var.cumsum(), marker="o")
        ax1.set_xlabel("主成分数")
        ax1.get_ylabel("累積寄与率")
        ax1.set_title("主成分の累積寄与率")
        st.pyplot(fig1)

        # 二次元プロット
        if X_pca.shape[1] >= 2:
           df_pca = pd.DataFrame(X_pca[:,:2], columns=["PCA1", "PCA2"])
           fig2, ax2 = plt.subplots()
           sns.scatterplot(x="PC1", y="PC2", data=df_pca, ax = ax2)
           ax2.set_title("主成分空間の二次元プロット")
           st.pyplot(fig2)           

elif page == "統計用語の解説":
    st.subheader("基礎統計量")
    st.markdown("""
**はじめに**
                
統計の基本的な指標は、たくさんの数字の集まり（データ）がどんな特徴を持っているのかを、一目で分かるように教えてくれる便利な道具です。

今回は、5人の友達グループの数学テスト（100点満点）の結果を例にして、それぞれの指標を見ていきましょう

**テストの点数： {70点, 60点, 80点, 10点, 80点}**       

**平均 - 全体をならした中心点**
                
「平均点」などでよく使います。これは、全てのデータを足し合わせて、データの個数で割った値のことです。データの中心がどこにあるのかを知るための、最も基本的な指標です。                

**【計算】**
                
(70点 + 60点 + 80点 + 10点 + 80点) ÷ 5 = 60点

このグループの数学の平均点は 60点 ということになります。

**ポイント**

平均は計算が簡単で分かりやすいですが、一つだけ極端に大きい、または小さい値（外れ値といいます）があると、そちらに引っ張られてしまうという弱点があります。今回の例では「10点」という低い点数に引っ張られて、平均点が少し下がっています。                               

**中央値 - 正真正銘のど真ん中**
                
中央値は、データを小さい順（または大きい順）に並べたときに、ちょうど真ん中に来る値です。データの真ん中を知る、もう一つの方法です。

**【計算】**
                
まず、点数を小さい順に並べ替えます。 {10点, 60点, 70点, 80点, 80点}

データは5つなので、真ん中（3番目）の値は… {10, 60, 70 , 80, 80}
                
中央値は **70点** です。

**ポイント**

中央値は、順番で真ん中を決めるので、平均と違って「10点」のような極端な値（外れ値）の影響を受けにくいという強みがあります。実際の感覚に近い「真ん中」を知りたいときにとても便利です。

**最頻値 (さいひんち) - 一番人気の値**

最頻値は、その名の通りデータの中で最も頻繁（ひんぱん）に出てくる値のことです。一番人気のある値、と考えれば分かりやすいかもしれません。

**【計算】**

{70点, 60点, 80点, 10点, 80点}

この中で、一番多く出てくる点数は…

80点 が2回出てきていますね。なので、最頻値は **80点** です。

 **ポイント**
                
**最頻値**は、アンケート結果で「一番多かった意見」を見たり、お店で「一番売れている商品のサイズ」を知りたいときなどに使われます。必ずしもデータの中心を示すわけではありませんが、「流行り」や「傾向」を知るのに役立ちます。

データの「散らばり」を見る指標たち

平均、中央値、最頻値でデータの「中心」が分かりました。でも、それだけではデータ全体を理解できたとはいえません。例えば、2つのクラスの数学の平均点がどちらも60点だったとします。

Aクラス: みんなが55点～65点の間にいる。

Bクラス: 0点や100点もいるけど、平均すると60点。

平均点は同じでも、中身は全然違いますよね？この「データの散らばり具合」を教えてくれるのが、範囲・分散・標準偏差です。

**範囲 - 散らばりの最大幅**
                
範囲は、データの最大値と最小値の差です。データの散らばり具合を一番シンプルに知る方法です。

**【計算】**
                
{10点, 60点, 70点, 80点, 80点}

最大値: 80点 / 最小値: 10点

80 - 10= 70点

範囲は 70点 となります。点数が70点の幅に散らばっている、ということが分かります。

**分散 と 標準偏差 - 散らばり具合の"平均"**
                
分散と標準偏差は、どちらも**「各データが平均値からどれくらい離れているか（散らばっているか）」を、より詳しく数値で表したもの**です。標準偏差は、分散の平方根（ルート）をとったもので、セットで使われます。

なぜこれらが必要？
                
範囲だけだと、最大値と最小値以外のデータがどう散らばっているか分かりません。みんなが平均点近くに固まっているのか、全体にバラバラなのかを知るために、分散や標準偏差が役立ちます。

【計算のイメージ】

**偏差を出す**
                
それぞれの点数が、平均点（60点）からどれだけ離れているかを計算します。

70点 → +10点

60点 → 0点

80点 → +20点

10点 → -50点

80点 → +20点

**偏差を2乗して、平均する（これが分散）**

プラスとマイナスをそのまま足すと0になってしまうので、全てをプラスにするために2乗します。そして、その平均をとります。
""")

    st.image("statistics/siki.png", width=1000)    

    st.markdown("""
**分散は 680 になります。**

ルートをとる（これが標準偏差）

分散は「2乗」してしまったので、単位が「点」ではなく「点²」という分かりにくいものになっています。これを元の単位「点」に戻すために、平方根（ルート）をとります。

680 ≒ 26.08

**標準偏差は 約26.1点 です。**

**ポイント**
            
標準偏差の数値が大きいほどデータは広く散らばっていて、小さいほどデータは平均値の周りに集まっている、と解釈できます。このグループの点数は、平均60点を中心に、だいたい±26点くらいの範囲に散らばっているんだな、というイメージが掴めます。            


""")
    st.subheader("回帰分析")
    st.markdown("""
はじめに：回帰分析ってなに？
                
**回帰分析**とは、ものすごくシンプルに言うと**「原因」と「結果」の関係性を調べて、未来を予測するための分析手法**です。

例えば、「気温（原因）が上がると、アイスの売上（結果）はどれくらい増えるんだろう？」という関係を、数式モデル（予測の線）で表そうとするのが回帰分析です。

今回はこの「気温とアイスクリームの売上」を例にして、分析結果を読み解くための重要用語を見ていきましょう！

**回帰係数 - 原因が結果に与える影響度**
                
回帰分析を行うと、（アイスの売上） = a ×（気温） + b のような予測式（回帰式）が得られます。

この式の a の部分が回帰係数です。これは、原因（気温）が1単位（1℃）変化したときに、結果（アイスの売上）がどれだけ変化するかを示しています。

【たとえば…】
                
分析の結果、回帰係数が「10」だったとします。

これは、「気温が1℃上がると、アイスの売上は10個増える」と予測できる、という意味になります。この数値が大きければ大きいほど、気温が売上に与える影響が大きい、ということですね。                

**説明変数 と 目的変数 - 分析の主役とゴール**
                
これは回帰分析の基本中の基本となる言葉です。

**目的変数**：予測したい「結果」のことです。今回の例では「アイスクリームの売上」がこれにあたります。**「従属変数」**とも呼ばれます。

**説明変数**：結果を説明するための「原因」のことです。「気温」や「広告費」などがこれにあたります。**「独立変数」**とも呼ばれます。

回帰分析は、「複数の説明変数（原因）を使って、目的変数（結果）をいかにうまく予測するか」というゲームだと考えると分かりやすいです。

**多重共線性 - 説明変数どうしが似すぎてない？**
                
多重共線性（通称：マルチコ）は、重回帰分析で特に注意が必要な問題です。これは、説明変数の中に、非常によく似た動きをするペアが存在する状態を指します。

【たとえば】
                
アイスの売上を予測するために、「気温」と「ビールの売上」を説明変数にしたとします。
                
しかし、よく考えると「気温が上がれば、ビールの売上も上がる」という強い関係がありますよね。

このように、説明変数どうしが強く関係していると、分析モデルは「アイスの売上を増やしているのは、気温なの？それともビールの売上なの？」と混乱してしまいます。その結果、個々の回帰係数が正しく計算できなくなり、分析結果全体の信頼性が失われてしまうのです。

 **ポイント**
                
分析を始める前に、説明変数どうしの相関関係をチェックし、似たような変数はどちらか一方を採用するなど、マルチコを避ける工夫が必要です。                                

**標準化回帰係数** - 影響力の強さを公平に比較！
                
「気温」と「広告費」では、単位が全く違いますよね（「℃」と「円」）。
                
そのため、前回説明した通常の回帰係数の大きさだけを見ても、「気温」と「広告費」のどちらがよりアイスの売上に強く影響しているのかを直接比較することはできません。

そこで登場するのが標準化回帰係数です。これは、すべての説明変数の単位の影響をなくし（＝標準化し）、純粋にどの変数が一番影響力が強いのかを比較できるようにした係数です。

**【どう見る？】**
                
この係数は、符号（プラスかマイナスか）を無視した絶対値が大きいほど、目的変数への影響力が強いことを意味します。

・気温の標準化回帰係数: 0.6

・広告費の標準化回帰係数: 0.3

この場合、「広告費よりも気温の方が、アイスの売上に対して2倍ほど強い影響力を持っている」と解釈できます。                

**ダミー変数 - 「はい/いいえ」を数字に変える**
                
回帰分析は基本的に数値データを扱いますが、「性別（男/女）」や「曜日（平日/休日）」のような、数値ではないカテゴリデータを分析に加えたいときもありますよね。

そんな時に使うのがダミー変数というテクニックです。これは、カテゴリデータを**「0」か「1」の数字に変換**した変数のことです。

【たとえば】
                
「休日かどうか」を分析に加えたい場合、休日なら「1」 / 平日なら「0」という新しい変数（ダミー変数）を作ります。

こうすることで、コンピュータは「1（休日）のときは、0（平日）のときと比べて、売上がどれくらい変化するのか」を計算できるようになります。これにより、分析の幅がぐっと広がります。

**p値 - その関係、本当に意味ある？**
                
回帰係数が「10」と出ても、「それって、**たまたまそうなっただけじゃないの？**偶然じゃない？」という疑いが残ります。

その疑いに答えてくれるのが p値 (p-value) です。これは、**「もし本当は関係ないとしたら、今回観測したような結果（か、それ以上に極端な結果）が偶然得られる確率」**を示します。

…ちょっと難しいですね。もっと簡単に言うと、

p値が小さいほど、「それは偶然とは考えにくい!この関係は意味がある（統計的に有意だ）!」と判断できる、ということです。

【どう判断するの？】
                
一般的に、p値が 0.05 (5%) 未満であれば、「偶然とは言えない、意味のある関係だ」と判断します。

・p値 < 0.05: 気温と売上には意味のある関係がありそうだ！

・p値 ≥ 0.05: この関係は、たまたまかもしれない...

つまり、回帰係数が大きくても、p値が大きければ「その影響度はアテにならないかも…」と考えるわけです。                

**決定係数 R² - モデル全体の当てはまり度**
                
回帰係数とp値で「気温と売上の関係性」が分かりました。では、作った予測式全体が、実際のデータをどれくらいうまく説明できているのでしょうか？

そのモデルの「当てはまりの良さ」や「予測の精度」を示すのが決定係数 (R-squared, R²) です。

【どう見る？】
                
決定係数は 0から1 の間の値をとり、1に近いほど、予測の精度が高いことを意味します。

例えば、

決定係数 R² = 0.85
                
これは「アイスクリームの売上のバラつきの 85% は、気温によって説明できますよ」という意味です。かなり精度の高い予測モデルと言えます。

決定係数 R² = 0.30
                
これは「売上のバラつきのうち、気温で説明できるのは 30% だけです」という意味です。売上には、気温以外の別の要因（曜日、天気、イベントの有無など）が大きく影響している可能性を示唆しています。
                
""")




#       streamlit run statistics/regression_tool.py 