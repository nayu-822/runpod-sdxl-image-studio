# RunPod SDXL Image Studio 開発計画

## Phase 4 追加修正の実装方針

- Migration 0007 は terminal evidence を優先し、通常の pending/ready 行を failed にしない。旧 0007 適用済み DB で安全に復元できない行は 0008 で `migration_status_ambiguous` / `ambiguous` に隔離し、自動再送しない。
- ComfyUI の modern job status は Adapter 内で `RemotePromptStatus` に変換する。modern endpoint の明示的な 404/405 の場合に限り queue/history へ fallback し、timeout・5xx・未知 payload は `UNAVAILABLE` として DB 状態を保持する。
- prompt ID mismatch の手動解決は Queue detail から Generation 側、Job 側、手入力を選択し、Generation / Job の prompt ID、status、submission state、audit、claim を同一 transaction で更新する。`/prompt` の再送は行わない。
- `execution_interrupted` の Recovery は Generation / Job の `cancelled_at`、`completed_at`、status、audit を同一 transaction で更新し、terminal state を上書きしない。
- 手動 prompt 解消で queued に戻す場合は、Generation / Job の terminal 日時を NULL にし、既存の cancel request は無条件に消去しない。
- Queue entry がない旧 Job の remote `NOT_FOUND` は `reconciliation_grace_seconds`（Job.updated_at → Job.created_at → Generation.updated_at → Generation.created_at）で判定し、期限超過時のみ `reconciliation_prompt_missing` の atomic failure とする。

## フェーズ4の実装状況

永続 FIFO Queue、単一 worker の lease、Random/連番 batch seed、キャンセル確認、単体 retry、failed-only batch retry、起動時・定期 reconciliation、Queue filter、0004/0005/0006/0007 migration、Gradio二重操作防止、Fake/Mockによる自動テストを実装済みです。キャンセル結果は `CANCELLED`、`COMPLETED`、`FAILED`、`IN_PROGRESS`、`NOT_FOUND`、`UNAVAILABLE` の typed outcome で扱い、history の `execution_interrupted` をキャンセル確定として処理します。

送信状態は `ready → submitting → submitted` または `ambiguous` です。`cancel_requested` はキャンセル要求の保存状態であり、ComfyUI 側の停止確認後にのみ `cancelled` へ遷移します。prompt ID を持つ Job は再送信せず、`IN_PROGRESS` と `UNAVAILABLE` は失敗へ変更しません。`NOT_FOUND` のみ grace 期間後の失敗候補です。

0005/0006では既存DBの状態不一致を terminal state と Artifact の証拠に基づいて補正し、Generation と Job の prompt ID 不一致は `migration_prompt_id_mismatch` として Queue entry ごと `ambiguous` に隔離します。0005/0006適用済みDBには0007後続migrationを適用し、cancel requestだけで `cancelled` へ遷移させず、復元できない状態を監査可能に保持します。既存の画像、Artifact、履歴、Presetは削除しません。

Queue 並べ替え、複数 worker/GPU、自動 retry、ComfyUI汎用workflow editor、LoRA学習は未実装です。

## 方針

機能を小さなフェーズに分け、各フェーズで動作確認可能な状態を維持する。最初から高度なワークフロー編集機能を作らず、固定された安全な workflow template を利用して、日常的な画像生成操作を先に完成させる。

また、追加希望された 1〜29 機能はすべてスコープに含める。ただし実装順は、運用上の重要度と依存関係に従って段階的に進める。

## フェーズ0: リポジトリ・設計基盤

### 目的

実装開始前に、プロジェクト構成、品質基準、設定方式を確定する。

### 対応

- `src` レイアウト
- `pyproject.toml`
- `.gitignore`
- `.env.example`
- README
- AGENTS
- CODING_RULES
- 詳細設計仕様書
- GitHub Actions
- ruff / mypy / pytest
- package 名・アプリ名の決定

### 完了条件

- ローカル CPU 環境で import とテストが実行できる
- 秘密情報や生成画像が Git 対象外になっている
- 文書間の優先順位が明確である

## フェーズ1A: ComfyUI接続・能力取得基盤

### 目的

Gradio から Application Service を経由して ComfyUI の稼働状態と利用可能な生成パラメータを取得できる状態にする。

### 完了状況

- `/system_stats` / `/object_info` の HTTP 取得
- checkpoint、VAE、sampler、scheduler、LoRA、upscaler の解析
- 接続状態と能力情報の Gradio 表示
- 手動の接続確認・一覧再読込
- fixture を使った単体・統合テスト

画像生成、`/prompt`、WebSocket、ワークフロー、SQLite、画像保存はフェーズ1B以降で実装する。

## フェーズ1B: 最小画像生成基盤

### 目的

RunPod 上で Gradio から ComfyUI へ txt2img を依頼し、画像を表示・保存できるようにする。

### 対応

- Gradio Blocks の最小 UI
- モバイル向けレスポンシブ CSS
- ComfyUI のヘルスチェック
- checkpoint 一覧取得
- sampler / scheduler 一覧
- positive / negative prompt（正・負のプロンプト）
- seed
- width / height
- steps / CFG
- 解像度プリセット
- 固定 txt2img workflow template
- `/prompt` 送信
- WebSocket 進捗
- `/history` による完了確認
- 出力画像取得
- ローカル保存
- SQLite の generation / job テーブル
- PNG metadata と sidecar JSON
- 失敗状態保存
- 生成前チェックの基礎
- ブラウザ再接続時の基本状態復元

### 対象外

- 複数 LoRA
- Google Drive
- 高度なアップスケール
- プリセット
- 外部画像 metadata 読込
- 詳細なシステム状態画面

### 完了条件

- RunPod Proxy 経由でスマートフォンから生成できる
- ブラウザ切断後も生成が継続する
- 再読込後に履歴を表示できる
- seed を含む生成条件が保存される
- 同じ snapshot から同条件再生成できる

## フェーズ2: モデル・複数 LoRA・LoRA 補助情報

### 目的

checkpoint と複数 LoRA を UI から安全に選択できるようにする。

### 対応

- checkpoint カタログ
- VAE カタログ
- LoRA カタログ
- 一覧再読込
- LoRA 行の追加・削除
- model strength / clip strength（モデル強度 / CLIP 強度）
- 最大 LoRA 数
- 重複 LoRA 防止
- LoRA 対応 workflow mapping
- 選択内容の metadata 保存
- 不足モデルの復元時警告
- モデル検索・絞り込み
- スマホ向け LoRA card UI
- LoRA カテゴリ
- LoRA お気に入り
- 最近使った LoRA
- LoRA トリガーワード
- LoRA 推奨強度
- LoRA 推奨モデル
- LoRA プレビュー画像

### 完了条件

- 複数 LoRA を順序付きで適用できる
- 強度を個別指定できる
- 同条件再生成時に LoRA の順序と強度が復元される
- 許可ディレクトリ外のファイルが一覧へ出ない

## フェーズ3: 履歴・検索・再生成・プリセット

### 目的

日常利用に必要な再利用操作を整える。

### 対応

- 日付別履歴
- Gallery のページング
- generation 詳細
- 同条件再生成
- seed 固定 / ランダム / 前回再利用切替
- 設定をフォームへ復元
- 一部条件変更後の派生生成
- お気に入り
- メモ
- 生成プリセット
- prompt プリセット
- LoRA プリセット
- アップスケールプリセット
- プリセットのスキーマバージョン
- 元 generation との関連付け
- プロンプト差分表示
- 実使用 seed のコピー
- 最近使った設定
- 履歴検索（モデル / LoRA / seed / prompt / 種別 / お気に入り / 解像度 / 成功失敗）

### 完了条件

- 過去画像からワンタップで設定を復元できる
- 同条件再生成と編集後生成を区別できる
- スマートフォンから履歴を実用的に閲覧できる
- 条件検索で過去画像を見つけられる

## フェーズ4: 生成キュー・バッチ生成・ジョブ管理強化

### 目的

複数条件の連続実行と、非同期処理の見える化を行う。

### 対応

- 生成キュー
- ジョブ一覧
- キュー順序表示
- キャンセル
- 再試行
- 失敗のみ再実行
- バッチ生成
- ランダム seed / 連番 seed
- ジョブ進捗表示
- stale job の復旧
- 再起動後 reconciliation

### 完了条件

- 複数ジョブを順番に実行できる
- スマホから予約投入できる
- ブラウザを閉じても進捗を後から確認できる

## フェーズ5: アップスケール

### 目的

生成済み画像から再現可能なアップスケール画像を作成する。

### 対応

- 直前画像のアップスケール
- 履歴画像のアップスケール
- 親 generation の関連付け
- 画像アップスケール用ワークフロー
- latent upscale / hires fix 用ワークフロー
- アップスケーラーモデルのカタログ
- 倍率または最終解像度
- denoise
- 同じ seed / prompt / model / LoRA の復元
- 元画像と比較表示
- 通常生成と別フォルダ保存
- アップスケール metadata
- アップスケール方式の明示的分離
- 出力サイズの事前表示
- 推定負荷表示
- 上限ガード

### 完了条件

- 元画像を上書きしない
- 親子関係が履歴で確認できる
- metadata から条件が完全復元される
- アップスケール固有条件も保存される

## フェーズ6: 外部画像 metadata インポート

### 目的

アプリ外または過去に生成された画像から条件を読み取り、アップスケールまたは再生成できるようにする。

### 対応

- PNG metadata パーサー
- ComfyUI prompt metadata パーサー
- アプリ sidecar JSON parser
- 画像アップロード
- 読取内容のプレビュー
- checkpoint / LoRA 存在確認
- 未解決項目の手動マッピング
- 安全な設定変換
- 読込元 metadata の原文保存
- スキーマ移行
- インポート画像の hash

### 完了条件

- metadata を実行前に確認できる
- 不足モデルがある状態で誤実行しない
- 任意 workflow や任意コードが metadata から実行されない
- sidecar JSON の読み込みに対応する

## フェーズ7: Google Drive 保存・再同期・容量可視化

### 目的

生成結果を日付別・種別別に Google Drive へ安全に保存する。

### 対応

- rclone アダプター
- 接続確認
- 日付フォルダ
- `generated/`
- `upscaled/`
- `manifests/`
- 画像と JSON の copy
- 転送進捗
- 再試行
- pending / syncing / synced / failed（待機中 / 同期中 / 同期済み / 失敗）
- 起動時の未同期検出
- 手動再同期
- 日次 manifest JSONL
- 保存先設定
- ローカル容量表示
- 未同期容量表示
- 同期済みキャッシュの削除候補表示

### 完了条件

- 画像と metadata が同じ日付・種別フォルダへ保存される
- 同期失敗時もローカル画像が残る
- 再試行できる
- `rclone sync` を使用しない
- Asia/Tokyo の日付でフォルダ分けされる

### 実装状況

フェーズ7を実装済みです。rcloneの引数配列Adapter、SQLiteの`drive_sync_records`／`drive_sync_jobs`／`drive_manifest_jobs`、
単一Workerのlease・heartbeat・stale復旧、生成完了後のenqueue、起動時のbounded discovery、手動retry／resync、
Asia/Tokyoの日付別manifest、destination snapshot、転送中progress/PID/log、ローカル容量・未同期容量・削除候補表示、同期・設定UIを追加しました。
画像とsidecar JSONは検証後に`copyto`で順に保存し、partial failureでもローカルを削除しません。
`rclone sync`、remote削除、複数Worker、自動retry、Driveからのdownloadは実装していません。
0014はmanifest再構築要求の追加だけを行い、0013以前のGeneration／Artifact／MetadataImportデータを変更しません。

## フェーズ8: モバイル UI 改善

### 目的

スマートフォンを主な操作端末としても不便がない状態にする。

### 対応

- 1 カラム基調
- 生成操作の固定表示
- prompt editor 改善
- LoRA カード
- 高度な設定のアコーディオン
- 生成中ステータス card
- 大きなタップ領域
- 画像比較
- 履歴フィルタ
- モバイル viewport テスト
- low bandwidth 時の thumbnail
- 画面復帰時の状態再取得
- 最近使ったモデル / LoRA / プリセット表示

### 完了条件

- 横スクロールなし
- 主要操作が片手でも可能
- 生成状況と選択画像が明確
- 大きい元画像を履歴一覧で直接読み込まない

### 実装状況

フェーズ8を実装済みです。生成画面をスマートフォンでは1カラム、デスクトップでは2カラムへ配置し、Positive/Negative promptの入力高さ、
LoRAカード、サイズ・Seed入力、高度な設定Accordion、バッチAccordion、stickyな生成ボタンを追加・整理しました。重要なボタンは44px以上の
tap targetを確保し、safe area、320pxから1024px以上までのresponsive breakpoint、focus outlineを共通CSSへまとめています。

生成中のstatus cardは`GenerationQueueService`と既存履歴Serviceから状態を読み取り、5秒間隔のbounded poll、`demo.load`による初期取得、
enqueue後の再取得でGeneration ID、Queue position、進捗、現在処理、完了Artifactを表示します。poll障害時は最後の表示を保持して安全な警告だけを表示し、
DBの状態遷移は変更しません。最近使ったcheckpoint、VAE、LoRA、Presetは既存`RecentSettingsService`とPreset handlerを再利用し、適用操作だけで生成を開始しません。

履歴一覧はthumbnail Artifactだけを使用し、thumbnail欠損時は軽量placeholderを表示します。原寸画像は詳細表示・完了結果表示のService経路だけで復元します。
履歴filter、Queue、Drive同期、metadata、Preset、upscale比較の既存機能はresponsive classを適用して維持し、LoRAのState・表示行・順序は既存handlerの同一出力で更新します。

Phase 8ではDB schema、migration、Generation/Job/Queue/DriveのDomain仕様を変更していません。Playwrightのviewportテストは
`.[browser]` optional dependencyを使い、CIでは一時SQLiteを使うローカルGradioを起動して実行します。ローカルでは
`IMAGE_STUDIO_BROWSER_URL`で起動済みURLを指定できます。テスト対象は320x568、375x812、390x844、430x932、768x1024、1280x800です。
Safari/Chromeのbackground復帰、ソフトキーボード表示中のsticky操作は未実施の手動確認です。

## フェーズ9: システム状態・エラー履歴・生成前チェック強化

### 目的

運用中の状態把握と、失敗時の復旧容易性を高める。

### 対応

- システム状態画面
- ComfyUI 接続状態
- GPU 名
- VRAM 使用量
- 生成キュー表示
- RunPod ディスク残量
- Google Drive 接続状態
- モデル数
- LoRA 数
- 最終同期時刻
- 生成前チェック
- checkpoint 存在確認
- LoRA 存在確認
- 空き容量確認
- 必要 custom node 確認
- エラー履歴
- 詳細ログへの導線

### 完了条件

- 実行前に主要な失敗要因を検知できる
- エラーの原因を後から追跡できる
- システムの健康状態をスマホから確認できる

### Phase 9 実装状況

Phase 9を実装済みです。既存の`ComfyUIService`、`GenerationQueueService`、
`DriveSyncService`、capability validationを集約し、`SystemHealthService`が
ComfyUI、GPU/VRAM、生成Queue、ローカルディスク、Google Drive、モデル数を
1回のスナップとして提供します。Systemタブは`demo.load`と手動更新のみで
更新し、リアルタイム連続pollは行いません。

Generation enqueue前には、永続化より前に`PreflightResult`を返す共通チェックを実行し、
ComfyUI接続、checkpoint、LoRA、VAE、upscaler、sampler、scheduler、required node、
ディスク残量を確認します。Google Driveの同期状態はwarningとし、生成のハードストップにはしません。
ワーカー側の最終validationは引き続き実行します。

Systemに紐づかない障害も扱うため、sanitizedなappend-only`system_error_events`テーブルと
Alembic `0015_phase9_system_error_events`を追加しました。API key、token、rclone secret、絶対パス、
プロンプ全文、生のtracebackは保存せず、最新100件までをSystemタブに表示しま。

`IMAGE_STUDIO_MIN_FREE_DISK_BYTES`が危陷閾値、`IMAGE_STUDIO_WARNING_FREE_DISK_BYTES`がwarning閾値です。
両者の大小関係は`Settings`で検証します。既存データを変更しない新規テーブルのみを使うため、
ダウングレードでは既存のGeneration/Job/Artifact/Queue/Driveデータに触れません。

Phase 9の自動確認は`tests/unit/test_phase9_system_health.py`で行い、FakeのComfyUI、Queue、Drive、
ディスクadapter、SQLiteを使用します。実GPU、実ComfyUI、実RunPod、実Google Driveは要求しません。

## フェーズ10: Stateless状態バックアップ・復元

### 目的

毎回新規PodをDeployし、利用後にTerminateする運用、またはVolume Disk 0GBのstateless運用でも、
SQLiteの状態を安全に引き継げるようにする。

### 対応

- SQLite backup APIによる状態snapshot
- Google Driveへのtimestamped state backupと`latest.json`ポインター
- backup hash、サイズ、SQLite integrity checkによる検証
- ローカルDBがない場合だけのatomic restore
- `NO_BACKUP`とremote取得不能の分離、およびrestore失敗時のfail-closed
- 復元後のGeneration / Job / Drive同期状態のstateless reconciliation
- dirty versionを用いたdebounced continuous backupとmanual clean backup
- worker停止後のbest-effort shutdown flush
- Generation、Preset、LoRA metadataの永続変更通知
- ローカルArtifact欠損時の履歴・upscale・Drive resync耐性

### 対象外

- model自動download
- 画像remote on-demand restore
- RunPod API
- 専用Docker image
- 自動Terminate

state backupのremote retentionと古いtimestamped backupの自動削除は未実装です。`rclone sync`や汎用remote削除APIは使用しません。

SQLite backup/restore、reconciliation、dirty version、LoRA通知、migration後のstartup順序はFake、SQLite、Alembicを使った
自動テストで確認します。実RunPod、実Google Drive、Volume Disk 0GBでの手動確認は別途必要です。

## フェーズ11: Google Driveモデルカタログ・選択取得

### 目的

Volume Disk 0GBまたは毎回Deployするstateless運用でも、必要なcheckpoint、LoRA、VAE、upscalerだけを
Google DriveからRunPodへ明示的に準備できるようにします。Remoteモデル一覧と、ComfyUIが現在利用できる
ローカルcapabilityは別の情報源として扱い、Remoteに存在するだけのモデルを生成画面へ表示しません。

### 対応

- `RemoteModelKind`、`RemoteModelEntry`、`RemoteModelCatalog`による型付きRemote catalog
- `rclone lsjson`の引数配列実行、拡張子・相対path・category root検証
- `model_transfer_jobs` SQLite migrationとpending/downloading/completed/failed/cancelled状態
- 同一Remote snapshotのactive transfer重複防止
- 一時ファイル、サイズ/hash検証、symlink/path containment確認、atomic replace
- ComfyUI capability refresh後のexact model visibility確認
- LoRA取得後の既存LoRA metadata catalog同期
- ブラウザ切断後も継続するModelTransferWorker、進捗・cancel・retry
- stateless restore時の未完了model transferの`stateless_restore_interrupted`終端化
- `StateSyncService.mark_dirty`への永続変更通知（progressは既存debounce経由）
- Generation画面とは分離した「モデル準備」タブと既存mobile viewport対応

### 設定

`RCLONE_REMOTE`を既存Drive同期と共有し、モデル領域は既定で`SDXLModels/`配下に分離します。
`IMAGE_STUDIO_REMOTE_MODEL_ENABLED`がfalseまたはcatalog取得不能でも、既にローカルにあるモデルによる生成は停止しません。
Remote catalogは選択・準備だけを提供し、Remote選択をGeneration設定へ自動適用したり、別モデルへ置換したりしません。

### 対象外

RunPod bootstrap/Docker image、RunPod API、自動Terminate、複数Pod、モデル自動更新・削除・LRU cache、
任意URL/Civitai download、過去画像のon-demand restore、img2img、inpainting、ControlNet、LoRA学習、
汎用workflow editorは対象外です。実RunPod、実ComfyUI、実Google Driveの手動確認は別途必要です。

## 将来候補

- img2img
- inpainting
- ControlNet
- 領域別プロンプト
- ADetailer 相当
- queue reorder の高度化
- 複数 workflow profile
- prompt ワイルドカード
- Dynamic Prompts
- X/Y/Z plot
- 自動 caption
- 画像評価・採否
- LoRA 作成プロジェクトとのモデル共有
- RunPod APIによるPod自動Deploy
- GPU自動選択
- 外部orchestration / scheduler
- 複数ユーザー対応

これらは初期設計へ直接組み込まず、Adapter と workflow template の拡張点だけ確保する。

## フェーズ2Aの状況

フェーズ2Aは完了しています。順序付き複数 LoRA 選択、モデル/CLIP 強度、checkpoint 内蔵または外部 VAE の選択、
固定ワークフローの対応付け、能力情報の事前検証、制限付き UI 編集を実装しました。
詳細は [PHASE_2A_IMPLEMENTATION.md](PHASE_2A_IMPLEMENTATION.md) を参照してください。

## フェーズ2Bの状況

フェーズ2Bは完了しています。SQLite による LoRA metadata カタログ、カテゴリ、お気に入り、
推奨情報、トリガーワード、プレビュー、検索、安全な失敗処理、生成フォームとの同期を実装しました。
LoRA プリセットは引き続き延期しています。

## フェーズ3Aの状況

フェーズ3Aの初期履歴基盤は完了しています。Generation、Job、Artifact レコードを Alembic 対応の
SQLite リポジトリへ永続化します。確定済み設定スナップショット、状態遷移、prompt ID、UTC タイムスタンプ、
画像ハッシュ、相対パスの成果物、WebP 履歴サムネイル、JSON sidecar、履歴ページング、詳細表示、復元、
派生再生成、お気に入り、メモを実装しました。未完了生成の基本復旧では prompt ID を確認しますが、
prompt の自動再送信は行いません。レビュー指摘への修正として、主 `image` Artifact と Generation / Job 完了を
同一 SQLite トランザクションで確定し、補助成果物の失敗を分離しました。既存 Artifact を優先する冪等な復旧、
一覧未取得と空一覧を区別する設定復元、LoRA 編集行の復元、再生成ボタンの排他制御も実装しています。
保守性改善として、prompt ID・主画像Artifact・完了・復旧・失敗状態の永続化エラーを分類し、
Generation と Job の failed 更新を `GenerationFailureRepository` の同一 SQLite トランザクションで確定します。
prompt ID の保存失敗時は自動再送信せず、失敗状態の保存自体に失敗した場合も元の生成エラーをログへ残します。

フェーズ3Aでは、バッチ、生成キュー、キャンセル、アップスケール、Google Drive 同期などのフェーズ3B以降の機能は実装しません。

フェーズ3Bには、高度な履歴検索（モデル、LoRA、seed、prompt、解像度、種別/状態の組み合わせ）、
プリセット、プリセットのスキーマ移行、プロンプト差分、最近使った設定へのショートカット、
seed コピー用ヘルパー、高度な派生ツリー UI を残しています。

## フェーズ3A保守性改善の完了

永続化Repositoryの自動補完を削除し、永続化を使わない構成または8 Repositoryを明示した構成だけを
GenerationServiceが受け付けるようにしました。Repository群は`GenerationPersistenceRepositories`へまとめ、
本番のcomposition rootから型付きで注入します。Start/Progressを含む原子的な責務は個別Repositoryに保持します。

## フェーズ3Bの完了状況

高度な履歴検索、検索用非正規化インデックス、Generation/Prompt/LoRA Preset、schema version 1のPayload、
最近使った設定、実使用seedのコピー用ヘルパー、親Generationとの差分、スマートフォン向け検索Accordionを
実装しました。Phase 3の完了条件である履歴からの設定復元、再生成と派生生成の区別、条件検索、Presetからの
設定復元、親Generationとの差分確認を満たします。

Phase 4へは、生成キュー、複数ジョブ、キャンセル、再試行、失敗のみ再実行、バッチ生成、stale jobの高度な
reconciliationを残します。Google Drive同期、複数ユーザー対応も
従来計画どおり後続フェーズです。

## フェーズ5の完了状況

フェーズ5の主要経路を実装済みです。完了済み画像Generationを親とする、非破壊で再現可能な画像アップスケールとLatentアップスケール、
アップスケール専用snapshotと`generation_upscale_settings` migration、一次Artifactの再検証、upscaler catalog、固定workflow、
ComfyUI input upload、送信前capability検証、出力寸法検証、upscaled保存先、キュー実行・再試行・復旧経路、比較用UIを追加しています。

ただし、実GPU・実ComfyUIを使った運用確認は未実施です。Fake/SQLite/Alembicによる自動テストで検証し、実環境では
capability不足時のfailed確定、再起動後のhistory復旧、誤寸法出力のfailed確定、元画像非上書き、UIの親画像選択と比較を手動確認する必要があります。

未実装として複数worker、queue並べ替え、自動retry、汎用workflow editorは引き続き対象外です。

## フェーズ6の完了状況

フェーズ6を実装済みです。PNG/WebPの安全な検証とcanonical PNG保存、SHA-256・寸法・形式の記録、ComfyUI既知prompt graphと
本アプリsidecar schema v1の候補解析、raw metadata保持、未解決項目のpreview、明示的なmodel mapping、SQLite `metadata_imports`
repository、0010/0011/0012 migrationを追加しました。workflow metadataはraw-onlyで保持し、実行・eval・exec・pickle・shellは行いません。sidecarのmalformed JSON、invalid UTF-8、unsupported schemaはcanonical画像保存と分離し、初期状態では画像upscaleだけを許可します。有効なPNG promptまたはsidecarを明示選択した場合は、無効sourceを`*_invalid_ignored`監査warningへ降格して生成復元とlatent upscaleを許可し、有効sourceがない場合は画像upscaleだけに制限します。保存先のparent symlinkはdata root外への解決を拒否し、legacy ambiguous rowはraw sourceから候補を再構築して自動選択しません。

previewからの生成条件適用は明示操作に限定し、外部画像はmetadataなしでも画像アップスケールへ利用できます。Latentアップスケールは
完全に解決済みのGenerationSettingsだけを受け付け、Queue投入を既存のSQLite transactionへ接続しました。sidecarのprompt空白、LoRAの
strength/order、VAEのnullを損なわず、PNG promptとsidecarの衝突はsource選択とhash確認をSQLiteへ保存します。ComfyUI graphはtarget
KSamplerからのmodel/CLIP/latent/VAEDecode接続を検証し、別branch・複数候補・execution-chain unknownは未解決にします。外部sourceのpath、hash、
寸法、PNG形式とcheckpoint、VAE、LoRA、sampler、scheduler、upscalerの能力はworker実行直前とretry時にも再検証し、変更時は
`metadata_import_source_changed` で失敗確定します。外部image/latentはuploadと固定workflowを経由し、既存promptのreconciliationでは再送せず、
retryはsource provenanceと `retry_of_generation_id` を保持した新規Generation/Jobとして冪等に作成します。

実ComfyUI、実GPU、実RunPod、実Google Driveを使った手動確認は未実施です。自動テストではFake/SQLite/Alembicでstrict parser、storage cleanup、
source selection、UIの排他再有効化、mapping、外部image/latentのworkerとreconciliation、source mutation、retry idempotency、migrationの候補消失を拒否する安全なdowngrade、
既存Queue経路を検証します。複数worker、Queue並べ替え、自動retry、
汎用workflow editorは対象外です。
## Phase 12: 前回セッション復元と安全なAuto-Terminate

Phase 12では、実行snapshotとは分離したversion付きのフォーム状態snapshotを
追加しました。生成またはbatch enqueueの成功後だけ保存し、起動時に非同期で
復元します。復元できない場合は最新のgeneration snapshotへ安全にfallbackし、
Phase 11のmodel-transfer serviceでcheckpoint、VAE、LoRA、任意のupscalerを
正確に準備します。不足モデルを暗黙に代替せず、起動時にComfyUI promptを
enqueueしません。

Pod lifecycleのsessionは現在の`RUNPOD_POD_ID`単位で管理します。
Auto-Terminateはopt-inで、enqueue成功後にarmし、grace/draining状態を経て、
generation、ComfyUI、model-transfer、Drive、manifest、state backupのreadinessを
確認します。cleanなfinal backupをawaitしてからself-onlyのRunPod DELETEを一度だけ
実行します。identity変更、競合、ambiguousなAPI応答、dirty stateではfail-closed
にします。migration `0017_phase12_session_lifecycle`はform stateとPod lifecycle
tableだけを作成し、downgradeでもそのtableだけを削除します。

## Phase 13: RunPod Docker・Bootstrap・本番Template

Phase 13では、既存アプリケーションをFresh RunPodへ配置するためのdeployment
基盤だけを追加しました。RunPod APIによるPod自動Deploy、複数Podのorchestration、
scheduler、Network Volume cache、model LRU、過去画像のon-demand restore、新しい
migrationは対象外です。

repositoryには固定した`runpod/comfyui:1.4.4-cuda12.8`をbase imageとして使い、
`/opt/image-studio-venv`へeditable installしたImage Studio、checksum検証済みの
rclone 1.74.2、deployment scriptを含めます。bootstrapはrclone Secretをtemporary
fileへ書き、0600を設定してatomic replaceし、内容をlogへ出さずremoteだけを検証します。
`/start.sh`を起動した後、固定local endpointへwall-clock deadline方式でreadinessを
確認します。既定timeoutは900秒で、各probeの`--max-time`は残り時間と5秒の小さい方へ
制限します。timeout=0は即時probeを最大1回だけ行います。

Image Studio起動後は、既定5秒のintervalで各probe完了後にComfyUIを再確認します。
成功時はfailure countを0へ戻し、12回連続失敗した場合だけpersistent failureとして
Image Studio、base processの順に停止してnon-zero終了します。ComfyUIのauto-restartは
行いません。

imageにはmodel、生成画像、SQLite state、credential、OAuth token、cookie、`.env`、
`rclone.conf`を含めません。State restore、Alembic upgrade、model preparation、
Drive同期、Phase 12のSafe Auto-Terminateはapplicationの責務です。本番Templateが
公開するHTTP portは7860だけで、healthcheckのstart periodは30分です。
deployment READMEにはSecret作成、image publish、Template設定、Fresh Pod確認、
troubleshootingを記載します。

Phase 13の検証では、Dockerfileと`.dockerignore`のpolicy、bootstrapのsyntaxとSecret
materialization、ComfyUI readinessとprocess supervision、GPU不要のimage smoke test、
Phase 10〜12のregression test、Alembic upgrade/downgrade互換性を確認します。
