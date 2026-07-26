# RunPod SDXL Image Studio 開発計画

## 方針

機能を小さなフェーズに分け、各フェーズで動作確認可能な状態を維持する。最初から高度なワークフロー編集機能を作らず、固定された安全な workflow template を利用して、日常的な画像生成操作を先に完成させる。

また、追加希望された 1〜29 機能はすべてスコープに含める。ただし実装順は、運用上の重要度と依存関係に従って段階的に進める。

## Phase 0: リポジトリ・設計基盤

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

## Phase 1A: ComfyUI接続・能力取得基盤

### 目的

Gradio から Application Service を経由して ComfyUI の稼働状態と利用可能な生成パラメータを取得できる状態にする。

### 完了状況

- `/system_stats` / `/object_info` の HTTP 取得
- checkpoint、VAE、sampler、scheduler、LoRA、upscaler の解析
- 接続状態と能力情報の Gradio 表示
- 手動の接続確認・一覧再読込
- fixture を使った単体・統合テスト

画像生成、`/prompt`、WebSocket、workflow、SQLite、画像保存は Phase 1B 以降で実装する。

## Phase 1B: 最小画像生成基盤

### 目的

RunPod 上で Gradio から ComfyUI へ txt2img を依頼し、画像を表示・保存できるようにする。

### 対応

- Gradio Blocks の最小 UI
- モバイル向けレスポンシブ CSS
- ComfyUI health check
- checkpoint 一覧取得
- sampler / scheduler 一覧
- positive / negative prompt
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

## Phase 2: モデル・複数 LoRA・LoRA 補助情報

### 目的

checkpoint と複数 LoRA を UI から安全に選択できるようにする。

### 対応

- checkpoint catalog
- VAE catalog
- LoRA catalog
- 一覧再読込
- LoRA 行の追加・削除
- model strength / clip strength
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

## Phase 3: 履歴・検索・再生成・プリセット

### 目的

日常利用に必要な再利用操作を整える。

### 対応

- 日付別履歴
- Gallery paging
- generation 詳細
- 同条件再生成
- seed 固定 / ランダム / 前回再利用切替
- 設定をフォームへ復元
- 一部条件変更後の派生生成
- お気に入り
- メモ
- generation preset
- prompt preset
- LoRA preset
- upscale preset
- preset schema version
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

## Phase 4: 生成キュー・バッチ生成・ジョブ管理強化

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
- stale job recovery
- 再起動後 reconciliation

### 完了条件

- 複数ジョブを順番に実行できる
- スマホから予約投入できる
- ブラウザを閉じても進捗を後から確認できる

## Phase 5: アップスケール

### 目的

生成済み画像から再現可能なアップスケール画像を作成する。

### 対応

- 直前画像のアップスケール
- 履歴画像のアップスケール
- parent generation relation
- image upscale workflow
- latent upscale / hires fix workflow
- upscaler model catalog
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

## Phase 6: 外部画像 metadata インポート

### 目的

アプリ外または過去に生成された画像から条件を読み取り、アップスケールまたは再生成できるようにする。

### 対応

- PNG metadata parser
- ComfyUI prompt metadata parser
- アプリ sidecar JSON parser
- 画像アップロード
- 読取内容のプレビュー
- checkpoint / LoRA 存在確認
- 未解決項目の手動マッピング
- 安全な設定変換
- 読込元 metadata の原文保存
- schema migration
- インポート画像の hash

### 完了条件

- metadata を実行前に確認できる
- 不足モデルがある状態で誤実行しない
- 任意 workflow や任意コードが metadata から実行されない
- sidecar JSON の読み込みに対応する

## Phase 7: Google Drive 保存・再同期・容量可視化

### 目的

生成結果を日付別・種別別に Google Drive へ安全に保存する。

### 対応

- rclone adapter
- 接続確認
- 日付フォルダ
- `generated/`
- `upscaled/`
- `manifests/`
- 画像と JSON の copy
- 転送進捗
- retry
- pending / syncing / synced / failed
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

## Phase 8: モバイル UI 改善

### 目的

スマートフォンを主な操作端末としても不便がない状態にする。

### 対応

- 1 カラム基調
- sticky generate action
- prompt editor 改善
- LoRA card
- advanced accordion
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

## Phase 9: システム状態・エラー履歴・生成前チェック強化

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

## 将来候補

- img2img
- inpainting
- ControlNet
- regional prompting
- ADetailer 相当
- queue reorder の高度化
- 複数 workflow profile
- prompt wildcard
- Dynamic Prompts
- X/Y/Z plot
- 自動 caption
- 画像評価・採否
- LoRA 作成プロジェクトとのモデル共有
- GDrive からのモデル取得
- RunPod API による Pod lifecycle
- 複数ユーザー対応

これらは初期設計へ直接組み込まず、Adapter と workflow template の拡張点だけ確保する。

## Phase 2A status

Phase 2A is complete: multiple ordered LoRA selection, model/CLIP strengths, checkpoint-internal or external VAE selection, fixed workflow mapping, capability prevalidation, and bounded UI editing are implemented. See [PHASE_2A_IMPLEMENTATION.md](PHASE_2A_IMPLEMENTATION.md).

Phase 2B remains for LoRA metadata and catalog features such as trigger words, categories, favorites, recommendations, previews, search, and presets.
