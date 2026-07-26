# RunPod SDXL Image Studio

RunPod GPU Pod 上で ComfyUI を画像生成バックエンドとして使用し、Gradio からスマートフォンでも快適に操作できる個人向け SDXL 画像生成アプリです。

## 目的

- RunPod 上の GPU と ComfyUI を利用して SDXL 画像を生成する
- PC・スマートフォンから同じ操作画面へアクセスする
- 生成条件、seed、モデル、LoRA、ワークフローを再現可能な形で保存する
- 過去画像の検索・再生成・派生生成・アップスケールを簡単にする
- 生成済み画像または画像内メタデータからアップスケールを実行する
- 完成画像とメタデータを Google Drive へ日付別に保存する
- ComfyUI の複雑なノード操作を日常利用時には意識せずに済むようにする

## 主な機能

### 1. 基本画像生成

- Positive / Negative Prompt
- Seed の自動生成・固定・再利用
- Width / Height
- Steps
- CFG
- Sampler
- Scheduler
- Batch size / Batch count
- SDXL checkpoint 選択
- VAE 選択
- Clip skip 相当設定（対応ワークフローのみ）
- 複数 LoRA の選択
- LoRA ごとの model strength / clip strength
- 解像度プリセット
- ランダム seed / 固定 seed / 前回 seed 再利用
- バッチ生成

### 2. 生成結果管理

- 生成画像のプレビュー
- スマートフォン向け縦長カード表示
- 画像ごとの生成条件表示
- お気に入り・メモ・タグ
- 同条件再生成
- seed 固定で一部条件だけ変更
- 画像のダウンロード
- 画像の親子関係表示
- プロンプト差分表示
- 実際に使用した seed の表示・コピー

### 3. 履歴・検索

- 日付別履歴
- モデルで絞り込み
- LoRA で絞り込み
- seed で絞り込み
- プロンプト文字列で絞り込み
- 通常生成 / アップスケールで絞り込み
- お気に入りで絞り込み
- 解像度で絞り込み
- 生成成功 / 失敗で絞り込み
- 低解像度サムネイルによる高速一覧表示

### 4. プリセット

- 生成設定一式の保存
- Prompt / Negative Prompt プリセット
- モデル + LoRA 構成プリセット
- 解像度プリセット
- アップスケール設定プリセット
- 最近使ったプリセットの表示

### 5. LoRA / モデル管理

- モデル一覧の取得・再読込
- LoRA 一覧の取得・再読込
- LoRA の複数同時利用
- LoRA のカテゴリ管理
- LoRA のお気に入り
- 最近使った LoRA 表示
- LoRA トリガーワードの保存
- LoRA 推奨強度の保存
- LoRA 推奨モデルの保存
- LoRA プレビュー画像の表示
- 不足モデル / 不足 LoRA の警告

### 6. アップスケール

- 直前の画像を同じ生成条件と seed で高解像度化
- 過去画像を選択してアップスケール
- PNG metadata または sidecar JSON から生成条件を復元
- Latent upscale と image upscale の方式選択
- アップスケーラーモデル選択
- 倍率、denoise、最終解像度指定
- 出力解像度の事前表示
- 推定負荷表示
- 最大解像度・倍率ガード
- 元画像とアップスケール画像の比較表示
- 元画像とアップスケール画像の関連付け
- 通常生成とアップスケールを別フォルダへ保存

### 7. 同期・保存

- 生成した画像を日付別・種別別に Google Drive へ保存
- 通常生成とアップスケールを別フォルダへ保存
- sidecar JSON も併せて保存
- 画像ごとの同期状態表示
- 同期失敗時の再試行
- 日次 manifest 保存
- ローカル保存とクラウド保存の両立
- 容量表示・未同期容量表示

### 8. システム・運用

- 生成キュー
- 進捗表示
- キャンセル
- 再試行
- ブラウザ切断後も処理継続
- 再接続後の状態復元
- システム状態画面
- ComfyUI 接続確認
- GPU / VRAM / ディスク残量表示
- Google Drive 接続状態表示
- 生成前チェック
- エラー履歴

### 9. モバイル UI

- スマホ向け 1 カラム設計
- 高度設定の折りたたみ表示
- 生成ボタンの固定表示
- 大きめのタップ領域
- LoRA のカード表示
- 主要操作の上部集約
- 横スクロールを発生させない設計
- 最近使った設定の表示

## 想定構成

```text
スマートフォン / PC
        |
        | RunPod HTTP Proxy
        v
Gradio UI
        |
        v
Application Services
        |
        +--> ComfyUI API Adapter
        |       |
        |       v
        |   ComfyUI Workflow / GPU
        |
        +--> Metadata / History Service
        |
        +--> Model / LoRA Catalog
        |
        +--> Job Queue Service
        |
        +--> Google Drive Storage Adapter (rclone)
        |
        +--> SQLite
```

## 推奨技術

- Python 3.11 以上
- Gradio Blocks
- ComfyUI HTTP API / WebSocket
- SQLite + SQLAlchemy + Alembic
- Pydantic
- Pillow
- rclone
- pytest / ruff / mypy
- RunPod HTTP Proxy

## 保存構成

```text
Google Drive/
└── RunPodSDXLImageStudio/
    └── 2026-07-26/
        ├── generated/
        │   ├── 20260726_103512_ab12cd34.png
        │   └── 20260726_103512_ab12cd34.json
        ├── upscaled/
        │   ├── 20260726_104201_ef56gh78.png
        │   └── 20260726_104201_ef56gh78.json
        └── manifests/
            └── 20260726.jsonl
```

画像には可能な範囲で PNG metadata を埋め込み、同じ内容を sidecar JSON にも保存します。PNG metadata が欠落・削除された場合も sidecar JSON と SQLite から復元できる構成とします。

## 文書

実装前に次の順で確認してください。

1. `RUNPOD_SDXL_IMAGE_STUDIO_DESIGN_SPEC.md`
2. `DEVELOPMENT_PLAN.md`
3. `CODING_RULES.md`
4. `AGENTS.md`

## Phase 0 の状態

Phase 0 では、実装を安全に始めるためのプロジェクト基盤を提供します。

- `src` レイアウトの Python パッケージ
- `pydantic-settings` による型付き設定
- スマートフォン幅を考慮した最小 Gradio Blocks UI
- pytest / coverage / ruff / mypy の設定
- Python 3.11 / 3.12 用 GitHub Actions
- 後続フェーズ用の domain / service / adapter / persistence / UI 境界

Phase 0 の起動時には ComfyUI、データベース、GPU、rclone へ接続しません。画像生成、履歴、SQLite、Google Drive 同期などは後続フェーズで追加します。

## 開発環境のセットアップ

Python 3.11 以上を用意し、リポジトリのルートで次を実行します。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
copy .env.example .env  # Windows PowerShell
# cp .env.example .env  # macOS / Linux
```

設定値は `.env.example` をコピーした `.env` で変更します。設定読み込みだけではディレクトリ作成や外部接続は行われません。

## Phase 0 の確認

```bash
ruff format --check .
ruff check .
mypy src
pytest --cov=runpod_sdxl_image_studio --cov-report=term-missing
python -m runpod_sdxl_image_studio.app
```

または editable install 後に console script を使えます。

```bash
runpod-sdxl-image-studio
```

ブラウザで `http://127.0.0.1:7860` を開くと、Phase 0 の状態画面が表示されます。`share=True` は使用しません。RunPod での本番起動や ComfyUI 接続は後続フェーズで追加します。

## セキュリティ

- `.env`、API キー、rclone 設定を Git に含めない
- Gradio の認証または RunPod 側のアクセス制御を使用する
- ComfyUI の API を直接インターネットへ公開しない
- 任意ワークフロー JSON、任意パス、任意コマンドを無検証で実行しない
- モデル・LoRA・アップスケーラーの選択肢は許可されたディレクトリ内に限定する
- 外部画像 metadata を信頼しない
