# Numerai入門から自動化へ

## はじめに
Numeraiは実際の金融市場に近い匿名データを用いて機械学習モデルを提出・評価できるコンペであり、実務力と理論の融合に最適な学習場である。本ドキュメントでは、学びながら実装と自動化を進める手順をまとめる。

---

## ステップ概要

1. **STEP&nbsp;1**: 環境構築とアカウント作成  
2. **STEP&nbsp;2**: データ取得と理解  
3. **STEP&nbsp;3**: ベースライン（LightGBMなど）を動かす  
4. **STEP&nbsp;4**: 自作モデル（ML → LSTM）に置き換える  
5. **STEP&nbsp;5**: 提出 → 評価 → 改良  
6. **STEP&nbsp;6**: 継続提出 or 自動化  

---

## STEP 詳細

### STEP 1 — 初期セットアップ

| タスク | コマンド / 手順 |
|-------|----------------|
| Numerai アカウント作成 | https://numer.ai にアクセスして登録 |
| API キー発行 | **Account → API keys** で取得 |
| Python 環境構築 | `pip install numerapi pandas numpy scikit-learn lightgbm` |
| GitHub リポジトリ | `numerai-team` を作成しクローン |

### STEP 2 — データを理解する

* データダウンロード  
  ```python
  import numerapi, pandas as pd
  napi = numerapi.NumerAPI()
  napi.download_dataset("v4.3", unzip=True)
  ```
* `train.parquet` の読み込み  
  ```python
  df = pd.read_parquet("numerai_dataset/train.parquet")
  ```
* `df.describe()` で統計量を確認し、`feature_*` が特徴量、`target` が目的変数であることを把握。

### STEP 3 — ベースラインモデルを作る

```python
from lightgbm import LGBMRegressor
X = df.filter(like="feature_")
y = df["target"]
model = LGBMRegressor()
model.fit(X, y)
preds = model.predict(X)
```

1. `prediction_id` と `preds` を結合して CSV を作成  
2. `napi.upload_predictions("predictions.csv")` で提出  
3. ダッシュボードでスコアを確認

### STEP 4 — LSTM への置き換え

* **PyTorch** あるいは **Keras** で実装  
* `feature` 群を `(batch, seq_len, features)` へ reshape  
* `MSE` で損失計算、学習  
* 予測結果を CSV で提出

### STEP 5 — 改良と自動化

| 施策 | 例 |
|------|----|
| 特徴量選択 | 相関が高い feature のみ使用 |
| 正規化 | `StandardScaler`, `MinMaxScaler` |
| ハイパーパラメータ最適化 | Optuna, GridSearch |
| 毎週提出自動化 | GitHub Actions で `cron` 発火 → 学習 → 提出 |

### STEP 6 — ロギングと継続運用

* `notebooks/log.md` に提出履歴と学びを記録  
* スコア改善を可視化（例: README バッジ）  

---

## チーム開発スタイル（分担しないコツ）

* **画面共有＋ペアプロ**：交代でドライバー／ナビゲーター  
* **共同で README 編集**：学んだことを都度プッシュ  
* **週次レビュー**：提出スコアを一緒に確認して改善点をディスカッション  
* **Issue をメモ代わりに**：やること・振り返りを GitHub Issue に残す  

---

## 推奨フォルダ構成

```plaintext
numerai-team/
├── notebooks/
│   ├── baseline_lightgbm.ipynb
│   ├── lstm_model.ipynb
│   └── log.md
├── data/
│   ├── train.parquet
│   └── tournament.parquet
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── submit.py
├── predictions/
│   └── predictions.csv
├── README.md
└── .gitignore
```

---

## おわりに

Numerai では“高い瞬間的精度”より **継続的な提出と改善のプロセス** が重視される。  
友人と二人で手を動かしながら、**「学ぶ → 提出 → 振り返る」** を回し、最終的には週次自動化まで育てることが大事。  
