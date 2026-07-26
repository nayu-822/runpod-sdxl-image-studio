# RunPod SDXL Image Studio 詳細設計仕様書

- 文書版数: 2.0
- 対象: RunPod + ComfyUI + Gradio による SDXL 画像生成アプリ
- 想定利用者: 単一ユーザー
- 主利用端末: スマートフォンおよび PC

## 1. 背景

ComfyUI は高い柔軟性を持つ一方、スマートフォンから日常的に使用する場合、ノード編集、パラメータ位置、履歴管理、モデル・LoRA 切替が煩雑になりやすい。

本アプリは ComfyUI を画像生成エンジンとして利用しつつ、日常的に使用する操作を Gradio の専用 UI に集約する。ComfyUI の workflow は安全なテンプレートとして管理し、利用者は prompt、seed、モデル、LoRA、解像度などの意味のある設定だけを操作する。

## 2. ゴール

- RunPod 上で SDXL 画像生成を実行する
- スマートフォンから操作しやすい UI を提供する
- checkpoint と複数 LoRA を選択できる
- LoRA のカテゴリ、お気に入り、トリガーワード、推奨設定を扱える
- 生成結果をその場で確認できる
- 全生成条件を保存し、再現可能にする
- 履歴検索ができる
- プリセットを保存できる
- 生成キューを扱える
- 同一条件・seed からアップスケールできる
- 画像 metadata から条件を復元できる
- 日付別、通常生成・アップスケール別に Google Drive へ保存する
- 同期状態、ディスク容量、システム状態を確認できる
- ブラウザ切断やアプリ再読込に耐える

## 3. 非ゴール

初期版では以下を対象外とする。

- ComfyUI の汎用 workflow editor
- 不特定多数ユーザー向け SaaS
- 任意 custom node の UI 追加
- 任意 URL からのモデルダウンロード
- モデルのライセンス自動判定
- 自動的な NSFW 判定
- LoRA 学習
- 動画生成
- 複数 GPU 分散生成

## 4. 用語

- Generation: 通常生成またはアップスケールを含む1回の生成単位
- Job: 非同期処理の実行管理単位
- Snapshot: 生成時点の固定された設定
- Parent Generation: アップスケールや派生生成の元画像
- Workflow Template: アプリが許可した ComfyUI workflow JSON
- Sidecar Metadata: 画像と同名で保存する JSON
- Model Catalog: checkpoint、VAE、LoRA、upscaler の選択可能一覧
- Preset: 再利用可能な設定テンプレート

## 5. システム構成

```text
[Mobile / PC Browser]
          |
          | HTTPS via RunPod Proxy
          v
[Gradio UI]
          |
          v
[Application Services]
  |       |        |         |         |
  |       |        |         |         +--> [System Status Adapter]
  |       |        |         +------------> [GoogleDriveAdapter / rclone]
  |       |        +----------------------> [SQLite]
  |       +-------------------------------> [Local File Storage]
  +---------------------------------------> [ComfyUIAdapter]
                                              |
                                              v
                                          [ComfyUI]
                                              |
                                              v
                                            [GPU]
```

ComfyUI 自体のポートは、可能な限り外部公開せず、Gradio アプリから localhost または内部ネットワークで接続する。

## 6. コンポーネント

### 6.1 UI

画面候補:

1. 生成
2. 履歴
3. アップスケール
4. プリセット
5. モデル
6. 同期・設定
7. システム状態

### 6.2 Application Services

- `GenerationService`
- `UpscaleService`
- `GenerationHistoryService`
- `PresetService`
- `ModelCatalogService`
- `MetadataImportService`
- `StorageService`
- `DriveSyncService`
- `QueueService`
- `SystemStatusService`
- `PromptDiffService`

### 6.3 Domain

主要モデル:

- `GenerationSettings`
- `LoraSetting`
- `UpscaleSettings`
- `Generation`
- `GenerationArtifact`
- `GenerationJob`
- `WorkflowTemplate`
- `ModelReference`
- `SyncRecord`
- `Preset`
- `HistoryFilter`
- `SystemStatusSnapshot`

### 6.4 Adapters

- `ComfyUIAdapter`
- `WorkflowTemplateAdapter`
- `LocalStorageAdapter`
- `GoogleDriveAdapter`
- `PngMetadataAdapter`
- `ModelCatalogAdapter`
- `SystemStatusAdapter`

## 7. ユースケース

### 7.1 基本生成

1. ユーザーが checkpoint、prompt、LoRA、seed などを指定
2. 生成前チェックを実施
3. Generation と Job を作成
4. ComfyUI へ送信
5. 進捗表示
6. 完了後、画像・metadata を保存
7. 履歴へ追加
8. Google Drive 同期キューへ追加

### 7.2 過去画像から再生成

1. 履歴から画像を選択
2. 保存済み snapshot を読み出し
3. UI へ復元または直接再生成
4. 新しい Generation を作成

### 7.3 アップスケール

1. 元画像を選択
2. image upscale か latent upscale を選択
3. 出力サイズと推定負荷を表示
4. 実行
5. 親子関係つきで保存

### 7.4 外部 metadata から復元

1. 画像をアップロード
2. PNG metadata または sidecar JSON を解析
3. 対応 schema に変換
4. 不足モデルを警告
5. ユーザー確認後に再生成またはアップスケール

## 8. データモデル

### 8.1 Generation

主な項目:

```text
id: UUID
kind: standard | upscale | derived
status: pending | queued | running | completed | failed | cancelled
parent_generation_id: UUID | null
created_at: datetime
started_at: datetime | null
completed_at: datetime | null
settings_snapshot_json: JSON
workflow_template_id: str
workflow_template_version: str
comfy_prompt_id: str | null
favorite: bool
user_note: str | null
error_code: str | null
error_summary: str | null
```

### 8.2 GenerationArtifact

```text
id: UUID
generation_id: UUID
artifact_type: image | metadata | thumbnail | log
local_path: str
sha256: str
size_bytes: int
width: int | null
height: int | null
mime_type: str
created_at: datetime
```

### 8.3 LoraSetting

```json
{
  "name": "character/example.safetensors",
  "model_strength": 0.8,
  "clip_strength": 0.8,
  "order": 0,
  "trigger_words": ["char_a"],
  "recommended_model": "model_x.safetensors"
}
```

### 8.4 SyncRecord

```text
id
generation_id
status: pending | syncing | synced | failed
remote_image_path
remote_metadata_path
attempt_count
last_attempt_at
error_summary
```

### 8.5 Preset

```text
id
preset_type: generation | prompt | lora | upscale | resolution
name
description
payload_json
created_at
updated_at
```

### 8.6 HistoryFilter

```json
{
  "model": null,
  "lora": null,
  "seed": null,
  "prompt_text": null,
  "generation_kind": null,
  "favorite": null,
  "resolution": null,
  "status": null
}
```

## 9. GenerationSettings

推奨 schema:

```json
{
  "schema_version": 1,
  "positive_prompt": "",
  "negative_prompt": "",
  "seed": 123456789,
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "cfg_scale": 5.5,
  "sampler_name": "euler_ancestral",
  "scheduler_name": "normal",
  "checkpoint_name": "models/example.safetensors",
  "vae_name": null,
  "loras": [],
  "batch_size": 1,
  "batch_count": 1,
  "clip_skip": null,
  "workflow_template_id": "sdxl_txt2img",
  "workflow_template_version": "1.0"
}
```

seed がランダム指定の場合も、実行前に確定 seed を generation snapshot へ保存する。

Batch 生成では、各出力画像を別 Generation とする方式と、1 Generation に複数 Artifact を持つ方式がある。履歴・親子関係・個別アップスケールを簡単にするため、推奨は「Job 1件、出力画像ごとに Generation 1件」とする。

## 10. ComfyUI Workflow Template

### 10.1 テンプレート種類

- `sdxl_txt2img`
- `sdxl_txt2img_lora`
- `sdxl_image_upscale`
- `sdxl_latent_upscale`

### 10.2 Template Definition

```json
{
  "template_id": "sdxl_txt2img",
  "schema_version": 1,
  "workflow_version": "1.0",
  "workflow_file": "workflows/sdxl_txt2img.json",
  "required_node_classes": [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "EmptyLatentImage",
    "VAEDecode",
    "SaveImage"
  ],
  "bindings": {
    "checkpoint": ["4", "inputs", "ckpt_name"],
    "positive_prompt": ["6", "inputs", "text"],
    "negative_prompt": ["7", "inputs", "text"],
    "seed": ["3", "inputs", "seed"]
  }
}
```

node ID は template adapter 内部だけで使用する。

### 10.3 検証

生成前に以下を確認する。

- template JSON が読める
- required node が存在する
- binding path が存在する
- checkpoint が catalog に存在する
- LoRA が catalog に存在する
- sampler / scheduler が対応値
- 出力 node が存在する
- 必須 custom node が ComfyUI 側で利用可能

## 11. ComfyUI 通信

### 11.1 起動確認

- `/system_stats` または利用可能な health endpoint
- `/object_info`
- WebSocket 接続

### 11.2 生成

1. UI 入力を validation
2. seed を確定
3. Generation と Job を DB に保存
4. workflow を構築
5. `/prompt` へ送信
6. prompt ID を保存
7. WebSocket で進捗監視
8. 完了時に history 取得
9. 出力画像取得
10. 画像検証
11. metadata 作成
12. ローカル確定保存
13. DB completed
14. GDrive 同期キュー

DB 保存前に ComfyUI へ送信すると prompt ID と generation の関連が失われる可能性があるため、送信前に pending record を作成する。

### 11.3 復旧

- WebSocket 切断時に即失敗扱いにしない
- prompt ID がある場合は history を確認する
- アプリ再起動時に running / queued job を reconcile する
- ComfyUI history に結果があれば保存処理を再開する
- 結果がなく queue にもなければ stale としてユーザー確認対象にする

## 12. アップスケール

### 12.1 方式

#### Image Upscale

- 元画像を入力
- upscaler model で拡大
- 必要に応じて VAE encode + sampler
- 元画像の構図を維持しやすい

#### Latent / Hires Generation

- 親 generation の prompt、seed、checkpoint、LoRA を復元
- 高解像度 latent または二段階生成
- denoise により描画差分が生じる

UI は方式、倍率、最終サイズ、denoise を明示する。

### 12.2 同条件の定義

「同条件」には以下を含む。

- prompt
- negative prompt
- seed
- checkpoint
- VAE
- LoRA 順序と強度
- sampler
- scheduler
- CFG
- steps

アップスケール固有設定は追加情報として保存する。

### 12.3 負荷表示・上限ガード

ユーザーに以下を表示する。

- 元画像サイズ
- 倍率
- 出力予定サイズ
- 推定負荷（低 / 中 / 高）
- 上限超過の警告

制限対象:

- 最大幅
- 最大高さ
- 最大総ピクセル数
- 最大倍率

### 12.4 外部画像

外部画像 metadata からアップスケールする場合:

1. metadata を抽出
2. 対応 schema へ変換
3. 読取結果を表示
4. モデル存在確認
5. 不足項目を表示
6. ユーザー確定後に実行

埋め込まれた ComfyUI workflow をそのまま実行してはならない。

## 13. Metadata

### 13.1 Sidecar JSON

```json
{
  "schema_version": 1,
  "app": {
    "name": "runpod-sdxl-image-studio",
    "version": "0.1.0"
  },
  "generation": {
    "id": "uuid",
    "kind": "standard",
    "parent_generation_id": null,
    "created_at": "2026-07-26T01:35:12Z"
  },
  "settings": {},
  "upscale": null,
  "artifacts": {
    "image_sha256": "",
    "width": 1024,
    "height": 1024
  },
  "runtime": {
    "comfyui_prompt_id": "",
    "workflow_template_id": "sdxl_txt2img",
    "workflow_template_version": "1.0"
  }
}
```

### 13.2 PNG Metadata

PNG には JSON 全体または主要項目を埋め込む。ただし閲覧・加工ソフトが metadata を削除する可能性があるため、sidecar JSON を必須とする。

## 14. 保存先

### 14.1 RunPod ローカル

```text
data/
├── app.db
├── generations/
│   └── 2026-07-26/
│       ├── generated/
│       └── upscaled/
├── thumbnails/
├── jobs/
└── logs/
```

### 14.2 Google Drive

```text
remote:RunPodSDXLImageStudio/
└── YYYY-MM-DD/
    ├── generated/
    ├── upscaled/
    └── manifests/
```

日付は Asia/Tokyo で決定する。

### 14.3 ファイル名

```text
YYYYMMDD_HHMMSS_<generation-id-short>.png
YYYYMMDD_HHMMSS_<generation-id-short>.json
```

prompt やモデル名を直接ファイル名に含めない。

## 15. モバイル UI

### 15.1 生成画面

上から以下の順を推奨する。

1. checkpoint
2. positive prompt
3. negative prompt
4. LoRA
5. サイズ preset
6. seed
7. 生成ボタン
8. 生成中 status
9. 結果画像
10. action buttons
11. advanced settings

主要項目を上部へ、高度な設定を Accordion へ配置する。

### 15.2 画像 action

- 同条件再生成
- 設定を編集
- アップスケール
- seed をコピー
- お気に入り
- GDrive 再同期
- metadata 表示
- 親子画像を表示
- プロンプト差分を表示

### 15.3 CSS 方針

- モバイルでは 1 column
- desktop では設定と preview の 2 column を許可
- button min-height を確保
- dropdown の高さを制御
- Gallery thumbnail を軽量化
- fixed width を避ける
- prompt textbox を狭くしない
- sticky generate action を許可

## 16. システム状態

表示項目:

- ComfyUI 接続状態
- GPU 名
- VRAM 使用量
- 実行中ジョブ数
- キュー数
- RunPod のディスク残量
- Google Drive 接続状態
- モデル数
- LoRA 数
- 最終同期時刻

## 17. 生成前チェック

最低限以下を確認する。

- checkpoint が存在する
- LoRA が存在する
- ComfyUI へ接続できる
- 解像度が上限以内
- 必要なカスタムノードがある
- 保存先に空き容量がある
- batch count が許容範囲内
- アップスケール倍率が上限以内

## 18. エラー履歴

- 失敗した generation を一覧表示する
- ユーザー向けの簡潔なエラーを表示する
- 詳細ログファイルへの導線を持つ
- retryable / non-retryable を区別する

## 19. 設定

`.env.example` 候補:

```dotenv
IMAGE_STUDIO_ENV=development
IMAGE_STUDIO_HOST=0.0.0.0
IMAGE_STUDIO_PORT=7860
IMAGE_STUDIO_TIMEZONE=Asia/Tokyo

COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_WS_URL=ws://127.0.0.1:8188/ws
COMFYUI_OUTPUT_DIR=/workspace/ComfyUI/output
COMFYUI_TIMEOUT_SECONDS=30

IMAGE_STUDIO_DATA_DIR=/workspace/image-studio-data
IMAGE_STUDIO_DATABASE_URL=sqlite:////workspace/image-studio-data/app.db
IMAGE_STUDIO_WORKFLOW_DIR=/workspace/runpod-sdxl-image-studio/workflows

IMAGE_STUDIO_CHECKPOINT_DIR=/workspace/ComfyUI/models/checkpoints
IMAGE_STUDIO_LORA_DIR=/workspace/ComfyUI/models/loras
IMAGE_STUDIO_VAE_DIR=/workspace/ComfyUI/models/vae
IMAGE_STUDIO_UPSCALER_DIR=/workspace/ComfyUI/models/upscale_models

IMAGE_STUDIO_MAX_WIDTH=2048
IMAGE_STUDIO_MAX_HEIGHT=2048
IMAGE_STUDIO_MAX_PIXELS=4194304
IMAGE_STUDIO_MAX_BATCH_COUNT=8
IMAGE_STUDIO_MAX_LORAS=8
IMAGE_STUDIO_MAX_UPSCALE_FACTOR=4.0
IMAGE_STUDIO_THUMBNAIL_SIZE=512

RCLONE_REMOTE=
RCLONE_BASE_PATH=RunPodSDXLImageStudio
RCLONE_CONFIG=
```

## 20. セキュリティ

- Gradio 認証または RunPod のアクセス制御を使用する
- ComfyUI を直接外部公開しない
- model path は allowlist directory 配下に限定
- symlink 解決後の path を検証
- 外部 metadata を信頼しない
- workflow の任意実行を許可しない
- shell command の自由入力を許可しない
- prompt はログへ不用意に全文出力しない
- 秘密情報を DB や metadata に入れない
- 画像 upload は実体形式とサイズを検証する

## 21. 非機能要件

### 可用性

- ブラウザ切断で job が失われない
- アプリ再起動後に履歴が残る
- GDrive 失敗で画像が失われない

### 性能

- 履歴は thumbnail と paging を使用
- 元画像を一覧で一括読込しない
- workflow template は必要に応じて cache
- model catalog は手動更新または短時間 cache

### 再現性

- 実行設定 snapshot を保存
- seed を確定保存
- workflow template version を保存
- model reference を保存
- 画像 hash を保存

### 保守性

- UI と ComfyUI Adapter を分離
- workflow binding を設定化
- schema migration を用意
- 外部依存を interface 化

## 22. 推奨ディレクトリ

```text
runpod-sdxl-image-studio/
├── .github/workflows/
├── alembic/
├── scripts/
├── src/runpod_sdxl_image_studio/
│   ├── app.py
│   ├── config.py
│   ├── domain/
│   ├── services/
│   ├── adapters/
│   │   ├── comfyui/
│   │   ├── metadata/
│   │   ├── models/
│   │   ├── storage/
│   │   └── status/
│   ├── persistence/
│   ├── jobs/
│   └── ui/
│       ├── components/
│       ├── tabs/
│       └── styles/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── workflows/
│   ├── definitions/
│   └── templates/
├── AGENTS.md
├── CODING_RULES.md
├── DEVELOPMENT_PLAN.md
├── RUNPOD_SDXL_IMAGE_STUDIO_DESIGN_SPEC.md
├── README.md
├── .env.example
└── pyproject.toml
```

## 23. 受け入れ条件

初期リリースは以下を満たす。

- スマートフォンから RunPod Proxy 経由で利用できる
- SDXL checkpoint を選択できる
- 複数 LoRA と強度を指定できる
- 基本パラメータを指定して生成できる
- 結果画像を確認できる
- 生成条件と seed が保存される
- 同条件で再生成できる
- 履歴検索ができる
- プリセットが使える
- 生成キューが使える
- 過去画像または metadata からアップスケールできる
- 通常生成とアップスケールが日付別・別フォルダで GDrive に保存される
- 同期失敗時にローカル画像を失わない
- 横スクロールなしで主要操作が行える
- システム状態と生成前チェックを確認できる
