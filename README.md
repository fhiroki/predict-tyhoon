# predict-typhoon

気象庁から公開されている各データを用いて、本土に台風が上陸した数を予測するプロジェクトです。

## 各ファイル説明

`analysis.R`
- [台風発生数](https://www.data.jma.go.jp/fcd/yoho/typhoon/statistics/generation/generation.html)
- [本土への接近数](https://www.data.jma.go.jp/fcd/yoho/typhoon/statistics/accession/hondo.html)
- [北太平洋の海面水温平年差の推移](https://www.data.jma.go.jp/gmd/kaiyou/data/db/climate/glb_warm/npac.txt)

を用いて、重回帰分析とポアソン回帰分析を用いて解析した。

`get_sea_temp_heatmap.py`
- [北太平洋の月平均海面水温](https://www.data.jma.go.jp/gmd/kaiyou/data/db/kaikyo/monthly/wnpsst.html)

から、日本近海の月平均海面水温のヒートマップ画像をスクレイピングしてくるスクリプト。

`main.py`

`get_sea_temp_heatmap.py` で取得した画像をkerasで読み込める形にして、学習する.
