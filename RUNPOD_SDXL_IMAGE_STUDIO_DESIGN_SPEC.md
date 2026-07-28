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

- ComfyUI の汎用ワークフローエディター
- 不特定多数ユーザー向け SaaS
- 任意カスタムノードの UI 追加
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
- Workflow Template: アプリが許可した ComfyUI ワークフロー JSON
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

### 6.2 アプリケーションサービス

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

### 6.3 ドメイン

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

### 6.4 アダプター

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
3. 現在の checkpoint、VAE、LoRA 一覧を取得済みの場合だけ存在確認を行う
4. UI へ復元し、復元失敗時は再生成を開始しない
5. 再生成ボタンを処理開始直後に無効化し、成功・失敗・検証失敗のいずれでも復帰する
6. 新しい Generation を作成

一覧が未取得の場合は `None` として扱い、空の一覧とは区別する。未取得時は
「現在のComfyUI一覧を取得していないため、モデルの存在確認は行っていません。」と表示し、
存在確認を省略したまま暗黙に代替モデルを選択してはならない。

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

### 8.1 Generation（生成）

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

### 8.2 GenerationArtifact（生成成果物）

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

`image` は Generation の必須主成果物である。`metadata` と `thumbnail` は任意の補助成果物であり、
保存に失敗しても Generation と Job の成功を取り消さない。ただし `image` の保存・登録に失敗した場合は
生成を成功扱いにせず、安全なエラーコードを記録し、確定済みの画像ファイルは削除しない。

### 8.3 LoraSetting（LoRA 設定）

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

### 8.4 SyncRecord（同期記録）

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

### 8.5 Preset（プリセット）

```text
id
preset_type: generation | prompt | lora | upscale | resolution
name
description
payload_json
created_at
updated_at
```

### 8.6 HistoryFilter（履歴フィルター）

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

## 9. GenerationSettings（生成設定）

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

## 10. ComfyUI ワークフローテンプレート

### 10.1 テンプレート種類

- `sdxl_txt2img`
- `sdxl_txt2img_lora`
- `sdxl_image_upscale`
- `sdxl_latent_upscale`

### 10.2 テンプレート定義

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

- テンプレート JSON が読める
- 必須ノードが存在する
- 対応付けパスが存在する
- checkpoint がカタログに存在する
- LoRA がカタログに存在する
- sampler / scheduler が対応値
- 出力ノードが存在する
- 必須カスタムノードが ComfyUI 側で利用可能

## 11. ComfyUI 通信

### 11.1 起動確認

- `/system_stats` または利用可能なヘルスエンドポイント
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
11. ローカル確定保存
12. `image` Artifact、Generation 完了、Job 完了を同一 SQLite トランザクションで記録
13. metadata と thumbnail を任意の補助成果物として保存
14. GDrive 同期キュー

DB 保存前に ComfyUI へ送信すると prompt ID と generation の関連が失われる可能性があるため、送信前に pending record を作成する。
必須成果物または完了状態の DB 保存に失敗した場合はトランザクションをロールバックし、Generation と Job を成功扱いにしない。
画像ファイルは保持して復旧対象とし、同じ prompt の再送信は行わない。

### 11.3 復旧

- WebSocket 切断時に即失敗扱いにしない
- prompt ID がある場合は履歴を確認する
- アプリ再起動時に実行中 / 待機中のジョブを整合させる
- ComfyUI の履歴に結果があれば保存処理を再開する
- 既存の主 `image` Artifact を最初に確認し、存在すれば再ダウンロード・重複登録・sidecar/thumbnail の重複作成を行わず、完了状態だけを整合させる
- 復旧処理は何度実行しても同じ結果になるよう冪等にする
- 結果がなくキューにもなければ、古い状態としてユーザー確認対象にする

## 12. アップスケール

### 12.1 方式

#### 画像アップスケール

- 元画像を入力
- アップスケーラーモデルで拡大
- 必要に応じて VAE エンコード + sampler
- 元画像の構図を維持しやすい

#### Latent / Hires 生成

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

## 13. メタデータ

### 13.1 sidecar JSON

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

### 13.2 PNG メタデータ

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
10. 画像操作ボタン
11. 高度な設定

主要項目を上部へ、高度な設定を Accordion へ配置する。

### 15.2 画像操作

- 同条件再生成
- 設定を編集
- アップスケール
- seed をコピー
- お気に入り
- GDrive 再同期
- metadata の表示
- 親子画像を表示
- プロンプト差分を表示

### 15.3 CSS 方針

- モバイルでは 1 カラム
- desktop では設定とプレビューの 2 カラムを許可
- ボタンの最小高さを確保
- ドロップダウンの高さを制御
- Gallery のサムネイルを軽量化
- 固定幅を避ける
- prompt 入力欄を狭くしない
- 生成操作の固定表示を許可

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
- 再試行可能 / 再試行不可を区別する

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
- モデルパスは許可リストのディレクトリ配下に限定
- シンボリックリンク解決後のパスを検証
- 外部 metadata を信頼しない
- ワークフローの任意実行を許可しない
- シェルコマンドの自由入力を許可しない
- prompt はログへ不用意に全文出力しない
- 秘密情報を DB や metadata に入れない
- 画像 upload は実体形式とサイズを検証する

## 21. 非機能要件

### 可用性

- ブラウザ切断で job が失われない
- アプリ再起動後に履歴が残る
- GDrive 失敗で画像が失われない

### 性能

- 履歴はサムネイルとページングを使用
- 元画像を一覧で一括読込しない
- ワークフローテンプレートは必要に応じてキャッシュする
- モデルカタログは手動更新または短時間のキャッシュを使用する

### 再現性

- 実行設定 snapshot を保存
- seed を確定保存
- ワークフローテンプレートのバージョンを保存
- モデル参照情報を保存
- 画像 hash を保存

### 保守性

- UI と ComfyUI Adapter を分離
- ワークフローの対応付けを設定化
- スキーマ移行を用意
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
