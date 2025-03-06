# マイナビ × SIGNATE Student Cup 2019（SOTAチャレンジ）の実装コード

## 大会概要
このプロジェクトでは、[マイナビ × SIGNATE Student Cup 2019: 賃貸物件の家賃予測](https://signate.jp/competitions/264)に基づき、東京都内の賃貸物件の家賃価格を予測するモデルを構築します。

## データセット
### コンペで提供されたデータセット
- `train.csv`: 訓練用データセット（物件情報と賃料）
- `test.csv`: テスト用データセット（物件情報のみ）
- `sample_submit.csv`: 提出フォーマット
### 追加したデータセット
- `geoencoded_districts.csv`: 地理情報データ（緯度・経度）
- `Jika_2019.csv`: 2019年の地価公示データ

## 使用したライブラリ
- **Polars**: 高速なデータフレーム処理
- **Pandas**: データ分析と操作
- **NumPy**: 数値計算
- **Matplotlib**: データ可視化
- **LightGBM**: 勾配ブースティングモデル
- **scikit-learn**: 機械学習ツール
- **SHAP**: モデル解釈
- **Plotly**: インタラクティブな可視化

## 参考資料とリンク
- [SIGNATE StudentCup 2019 1位ソリューション](https://github.com/analokmaus/signate-studentcup2019/)
- [国土交通省 国土数値情報](https://nlftp.mlit.go.jp/)
- [東京都財務局 地価公示](https://www.zaimu.metro.tokyo.lg.jp/kijunchi/chikakouji/31kouji)
