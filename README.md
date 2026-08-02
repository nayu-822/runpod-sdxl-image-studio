# RunPod SDXL Image Studio

## Phase 4 追加修正: migration と prompt 状態の復旧

- `0007_phase4_terminal_state_repair` は、Generation / Job の status、完了・キャンセル時刻、Artifact、prompt ID、cancel request を証拠として評価します。通常の `pending/pending + ready + prompt IDなし + cancelなし` は `pending/ready` のまま保持します。
- 既に旧 `0007` を適用済みで、元の pending と断定できない行は `0008_phase4_recovery_correction` で自動再送対象に戻さず、`migration_status_ambiguous` として監査可能な `ambiguous` に隔離します。
- ComfyUI の状態確認は typed な `RemotePromptStatus`（`PENDING`、`IN_PROGRESS`、`COMPLETED`、`FAILED`、`CANCELLED`、`NOT_FOUND`、`UNAVAILABLE`）で扱います。`/api/jobs/{prompt_id}` が明示的に 404/405 の場合だけ `/queue` と `/history` を使い、timeout / 5xx では破壊的 fallback を行いません。
- prompt ID が不一致または送信結果不明の行は自動再送しません。Queue detail で Generation 側、Job 側、または手入力の prompt ID を選び、`prompt IDを紐付け` で両方を同一 ID に解決します。prompt が存在しないことを確認した場合だけ `failed` を明示確定できます。
- History の `execution_interrupted` は Generation と Job を一つの SQLite transaction で `cancelled` に確定します。既に `completed` / `failed` / `cancelled` の行は上書きしません。

## フェーズ4: 永続キュー・復旧・キャンセル

SQLite の FIFO キューへ検証済みの `Generation`、`GenerationJob`、Queue entry を同一トランザクションで保存し、アプリケーションプロセス内の単一 worker が ComfyUI への送信と実行を担当します。ブラウザを閉じてもキューと実行状態は DB に残ります。

### 送信状態と再送信防止

Queue entry は `ready → submitting → submitted` または `ambiguous` の状態を持ちます。`submitting` へ遷移するときに UUID の `submission_token` と UTC の `submission_started_at` を永続化し、その token を ComfyUI の `client_id` に使用します。`/prompt` 成功後は prompt ID、Generation、Job、Queue entry を同一 DB トランザクションで `submitted` にします。

prompt ID の保存に失敗した場合は `ready` に戻しません。結果不明の `submitting` は `ambiguous` として隔離し、自動再送信しません。ambiguous の手動確認後にのみ運用者が判断します。DB障害で ambiguous も保存できない場合、worker は fail-closed で停止します。

### 復旧と状態

起動時および稼働中に prompt ID を持つ `queued` / `running` / `submitting` / `ambiguous` を reconciliation します。結果は `IN_PROGRESS`、`COMPLETED`、`FAILED`、`CANCELLED`、`NOT_FOUND`、`UNAVAILABLE` に typed model として区別します。prompt ID がある Job は、どの結果でも `/prompt` へ再送信しません。`NOT_FOUND` のみ grace 期間経過後に監査用エラーコードで失敗扱いにします。`UNAVAILABLE` や `IN_PROGRESS` は状態を維持します。完了時は既存 Artifact を確認して冪等に主 Artifact と完了状態を更新します。

0004適用済み DB の状態補正は、terminal state と主画像 Artifact を優先し、Generation と Job の prompt ID 不一致を `migration_prompt_id_mismatch` として Queue entry ごと `ambiguous` に隔離します。0005/0006適用済み DBには後続の0007 migrationを適用し、cancel requestだけで `cancelled` にせず、復元できない行を監査可能な状態に保持します。既存画像、Artifact、履歴、Preset は削除しません。

### キャンセルと再試行

未claimの pending は `cancel_requested` 保存後に cancelled へ遷移できます。claim済み、submitting、queued、running は即時に terminal へせず、ComfyUI の対象 prompt の queue削除または interrupt 後、queue/history で停止を確認できた場合だけ cancelled にします。history の `execution_interrupted` は cancelled として扱い、完了・失敗が先に確定していた場合はその terminal state を維持します。確認不能・接続失敗時は元の状態と `cancel_requested` を保持します。completed/failed/cancelled へのキャンセルは冪等です。

単体 retry と failed-only batch retry は元 Generation/Batch との関連を持つ新規キュー項目です。同じ retry 要求を複数回受けても、0005 の NULL許容 partial unique index により既存結果を返し、新規項目を重複作成しません。Random seed、連番 seed、SQLite保存値の上限は `MAX_SEED = 2**63 - 1` に統一しています。

Queue 並べ替え、複数 worker/GPU、自動 retry、Phase 5アップスケール、外部画像metadata、Google Drive同期、汎用workflow editor、LoRA学習は今回の対象外です。

## フェーズ3A: 生成履歴・スナップショット・再生成

フェーズ3Aでは、SQLite による `Generation`、`GenerationJob`、
`GenerationArtifact` レコードを追加します。実使用 seed と順序付き LoRA 強度を含む
確定済みの生成設定を、スキーマバージョン付き JSON スナップショットとして保存します。
完了した画像は、`<IMAGE_STUDIO_DATA_DIR>/generations/YYYY-MM-DD/` 配下に、相対パスの画像、
サムネイル、UTF-8 の sidecar metadata 成果物として保存されます。
Queue・Completion・Failure は用途別のトランザクション Repository として分離し、失敗時は
Generation と Job を同一トランザクションで更新します。prompt ID 保存に失敗しても同じ要求を自動再送信しません。

履歴タブでは、サービス API を通じて日付・状態・お気に入り・種別によるページング、
詳細表示、設定復元、同条件の派生生成、お気に入り、2,000 文字のメモに対応します。
未完了レコードは prompt を再送信せずに確認できます。高度な検索、プリセット、
プロンプト差分、バッチ生成などのフェーズ3B機能は引き続き保留です。対応する Gradio
バージョンは `>=5,<6` です。

## フェーズ2B: LoRA メタデータカタログ

フェーズ2Bでは、LoRA メタデータ用のローカル SQLite カタログを追加します。既定のデータベースは
`<IMAGE_STUDIO_DATA_DIR>/database/image_studio.sqlite3` です。明示的なデータベース URL を使う場合は
`IMAGE_STUDIO_DATABASE_URL` を設定します。アプリケーションは起動時に Alembic のマイグレーションを
明示的に実行し、モジュールの import によるデータベースまたはファイルシステムへの副作用はありません。

LoRA 管理タブでは、能力情報の同期、検索、カテゴリ・お気に入り・不足項目による絞り込み、
メタデータ編集、使用回数順の並べ替え、安全な PNG/JPEG/WebP サムネイル変換、UUID 名の WebP
ファイル保存に対応します。トリガーワードは、利用者が明示的なトリガーワードボタンを押した場合に
限って positive prompt へ追加します。推奨強度は LoRA の選択が変わった場合に限って適用します。
使用統計は可能な範囲で収集し、成功した生成を失敗へ変更することはありません。

フェーズ2Bでは、外部メタデータサービス、Civitai、Google Drive、rclone、RunPod API へ接続しません。
対応する UI バージョンは引き続き Gradio 5 です。

マイグレーションコマンド:

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

フェーズ2A（複数 LoRA と VAE 選択の基盤）:
[PHASE_2A_IMPLEMENTATION.md](PHASE_2A_IMPLEMENTATION.md)

フェーズ1B の実装範囲と手動確認手順は [PHASE_1B_IMPLEMENTATION.md](PHASE_1B_IMPLEMENTATION.md) にまとめています。

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

- 正・負のプロンプト
- seed の自動生成・固定・再利用
- 幅 / 高さ
- ステップ数
- CFG
- サンプラー
- スケジューラー
- バッチサイズ / バッチ回数
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
アプリケーションサービス
        |
        +--> ComfyUI API アダプター
        |       |
        |       v
        |   ComfyUI ワークフロー / GPU
        |
        +--> メタデータ / 履歴サービス
        |
        +--> モデル / LoRA カタログ
        |
        +--> ジョブキューサービス
        |
        +--> Google Drive ストレージアダプター (rclone)
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

## フェーズ0の状態

フェーズ0では、実装を安全に始めるためのプロジェクト基盤を提供します。

- `src` レイアウトの Python パッケージ
- `pydantic-settings` による型付き設定
- スマートフォン幅を考慮した最小 Gradio Blocks UI
- pytest / coverage / ruff / mypy の設定
- Python 3.11 / 3.12 用 GitHub Actions
- 後続フェーズ用の domain / service / adapter / persistence / UI 境界

フェーズ0の起動時には ComfyUI、データベース、GPU、rclone へ接続しません。画像生成、履歴、SQLite、Google Drive 同期などは後続フェーズで追加します。

## フェーズ1Aの状態

フェーズ1Aでは、ComfyUI の状態と利用可能な能力を確認する基盤を追加しました。

- `httpx.AsyncClient` による ComfyUI HTTP クライアント
- `/system_stats` と `/object_info` の取得
- checkpoint / VAE / sampler / scheduler / LoRA / upscaler の一覧解析
- 欠損ノード、欠損フィールド、接続失敗、timeout の安全な処理
- Application Service 経由の状態集約
- 生成タブとシステムタブの状態表示
- 接続確認とモデル一覧の手動再読込
- 実 ComfyUI に接続しない fixture ベースの単体・統合テスト

CSSをUI構築側の `gr.Blocks` に保持するため、フェーズ1AではGradio 6未満を使用します。

ComfyUI は `COMFYUI_BASE_URL` で指定した URL から読み取ります。RunPod では通常、同一 Pod 内の `http://127.0.0.1:8188` へ接続する想定です。アプリ起動だけでは ComfyUI へ接続せず、「接続確認」または「モデル一覧を再読込」を押したときだけ通信します。

フェーズ1Aでは画像生成、`/prompt`、WebSocket、ワークフロー、画像保存、SQLite、Google Drive 同期はまだ実装していません。

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

## フェーズ0の確認

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

ブラウザで `http://127.0.0.1:7860` を開くと、フェーズ1Aの状態画面が表示されます。`share=True` は使用しません。RunPod での本番起動設定は後続フェーズで追加します。

## フェーズ1Aの手動確認

```bash
cp .env.example .env
python -m runpod_sdxl_image_studio.app
```

ブラウザで次を確認します。

1. 「生成」と「システム」タブが表示される
2. 初期状態が「未確認」、各一覧が「未取得」である
3. 「接続確認」で system stats と能力情報が更新される
4. 「モデル一覧を再読込」で Dropdown と LoRA 一覧が更新される
5. ComfyUI 停止時もアプリは落ちず、安全な日本語メッセージが表示される

テストおよび CI は実 ComfyUI、GPU、Google Drive を必要としません。

## セキュリティ

- `.env`、API キー、rclone 設定を Git に含めない
- Gradio の認証または RunPod 側のアクセス制御を使用する
- ComfyUI の API を直接インターネットへ公開しない
- 任意ワークフロー JSON、任意パス、任意コマンドを無検証で実行しない
- モデル・LoRA・アップスケーラーの選択肢は許可されたディレクトリ内に限定する
- 外部画像 metadata を信頼しない

## Phase 3B: 履歴検索とPreset

履歴タブでは、検索テキスト、checkpoint、VAE、LoRA、seed、解像度、status、Generation kind、
お気に入り、error code、親Generation、日付範囲を組み合わせて検索できます。LoRAを複数指定した場合は
「いずれかを含む（ANY）」または「すべてを含む（ALL）」を選べます。検索テキストはPositive/Negative
Prompt、メモ、モデル名、LoRA名、エラー概要を対象にし、SQLのbind parameterとLIKE escapeを使用します。
入力は大文字小文字を区別しない検索です。

PresetにはGeneration、Prompt、LoRAの3種類があります。Payloadはschema version 1の型付きJSONとして
保存し、Generation Presetはcheckpoint、VAE、解像度、サンプラー、seed mode、Prompt、LoRA強度を保持します。
Prompt Presetは置換・先頭追加・末尾追加、LoRA Presetは置換・末尾追加を選べます。Presetを適用するだけでは
生成は開始されません。checkpoint、VAE、LoRAが不足している場合は警告し、自動置換・自動削除は行いません。

最近使ったcheckpoint、VAE、LoRA、PresetはDBへ件数制限付きで問い合わせます。履歴詳細ではsnapshotに保存された
実使用seedをコピーでき、親Generationとの差分ではPromptの追加・削除・並び替えと、生成設定・LoRA強度を確認できます。
PromptはHTMLとして解釈せず、差分表示時にescapeします。

最近使ったcheckpoint/VAEは明示ボタンでcapability一覧に存在する値だけを反映します。最近使ったLoRAは
「LoRAへ追加」ボタンで末尾へ追加し、重複・上限超過・missingは拒否します。最近LoRAにはCatalogの推奨強度を
自動適用せず、現在のLoRA editorと同じmodel/CLIP strengthの初期値1.0を使用します。

検索用のcheckpoint/VAE/seed/解像度カラムと`generation_loras`はsnapshotから同じトランザクションで作成される
インデックスです。生成設定の復元には検索用データを使わず、常にsnapshotを使います。

Phase 3Bの履歴UIはoffsetページングに統一しています。検索実行時はpage=1へ戻り、前へ・次へでは
現在の検索条件を維持したままoffsetだけを変更します。Phase 3Bではcursor入力とcursor生成を持たず、
offsetだけを適用する設計です。checkpointとVAEはUIでは単一指定、statusとkindは複数指定に対応します。
Migration backfillで壊れたsnapshot行を検出した場合は、その行をskipしてwarningを出し、他の正常行のbackfillを継続します。

### Phase 3B DB migration

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

Presetと検索用インデックスはSQLiteの`data/database/image_studio.sqlite3`へ保存されます。Gradioは引き続き
5系（`gradio>=5.0.0,<6.0.0`）を使用します。
