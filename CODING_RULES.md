# RunPod SDXL Image Studio コーディングルール

- 文書版数: 2.0
- 対象: Python、Gradio、ComfyUI Adapter、シェルスクリプト、設定、テスト

## 1. 基本方針

- 可読性、保守性、安全性、再現性を優先する。
- UI と画像生成ロジックを分離する。
- ComfyUI 固有処理は Adapter に閉じ込める。
- 生成画像・生成条件を破壊的に上書きしない。
- ブラウザの一時状態を正としない。
- 長時間処理はジョブとして管理する。
- 実装と同時にテストを追加する。
- 不明な例外を握りつぶさない。

## 2. Python

- Python 3.11 以上を対象とする。
- `src` レイアウトを使用する。
- `from __future__ import annotations` を原則使用する。
- 公開 API には型注釈を付ける。
- パスは `pathlib.Path` を使用する。
- 日時は timezone-aware とし、保存は UTC、表示・日付フォルダは Asia/Tokyo とする。
- 内部 ID は UUID を使用する。
- 複雑な入出力は dataclass または Pydantic model を使用する。
- 状態値は Enum を使用する。
- JSON 設定・metadata には `schema_version` を持たせる。

## 3. 命名

- モジュール・関数・変数: `snake_case`
- クラス・例外: `PascalCase`
- 定数: `UPPER_SNAKE_CASE`
- bool: `is_`, `has_`, `can_`, `should_`, `enable_`
- `data`, `info`, `tmp`, `obj` のような曖昧名を広いスコープで使用しない
- ComfyUI node ID を意味のあるドメイン名として扱わない

## 4. レイヤー

```text
UI (Gradio)
  ↓
Application Services
  ↓
Domain / Job Management
  ↓
Adapters / Persistence / File Storage
```

禁止事項:

- Domain から Gradio を import する
- UI イベント内で workflow JSON を直接書き換える
- Repository から ComfyUI API を呼ぶ
- Adapter から UI コンポーネントを返す
- DB model をそのまま UI 表示 model として使用する

## 5. 推奨アダプター

- `ComfyUIClient`
- `WorkflowAdapter`
- `ModelCatalogAdapter`
- `ImageMetadataAdapter`
- `StorageAdapter`
- `GoogleDriveAdapter`
- `Clock`
- `HashService`
- `SystemStatusAdapter`

外部サービスやファイル形式に依存する処理は interface 経由で利用する。

## 6. 生成設定

`GenerationSettings` は最低限以下を型付きで保持する。

- prompt
- negative_prompt
- seed
- width
- height
- steps
- cfg_scale
- sampler_name
- scheduler_name
- checkpoint_name
- vae_name
- LoRA 設定一覧
- batch_size
- batch_count
- workflow_template_id
- workflow_template_version

検証例:

- width / height は対応する倍数へ限定
- 最大ピクセル数を設定
- steps、CFG、batch 数に上限を設定
- seed の範囲を検証
- LoRA 強度を設定範囲に制限
- LoRA 数に上限を設定
- LoRA 重複を禁止
- 空の checkpoint を拒否

## 7. Workflow

- workflow template は Git 管理可能な JSON とする。
- 実行時には template をコピーして必要項目だけ差し替える。
- node class、node ID、入力キーを template 定義と mapping によって管理する。
- 任意 node の追加を UI から許可しない。
- template には `template_id`、`schema_version`、必要 custom node を定義する。
- template 変更時には compatibility test を追加する。
- JSON の直接文字列置換は禁止する。
- 生成前に必須 node と入力を検証する。

## 8. ComfyUI Client

- 接続・キュー送信・進捗購読・履歴取得・画像取得を分離する。
- HTTP timeout と WebSocket timeout を設定する。
- retry は冪等性を考慮する。
- `/prompt` の二重送信を避ける。
- prompt ID を永続化してから監視を開始する。
- WebSocket 切断時は history API で状態を確認する。
- キャンセル操作は ComfyUI の queue 操作とアプリ DB 状態を整合させる。
- ComfyUI のエラー全文はログへ保存し、UI には安全な要約を表示する。

## 9. ファイル保存

- 一時ファイルへ保存し、検証後に `os.replace` する。
- ファイル名はアプリ側で生成する。
- 画像 SHA-256、ファイルサイズ、寸法、実体形式を記録する。
- 拡張子だけを信用しない。
- 元画像を上書きしない。
- 親画像と派生画像を別レコードにする。
- metadata JSON と画像の片方だけが確定した状態を残さない。
- 不完全保存は cleanup または recovery 対象として記録する。
- サムネイルは原寸画像と別管理する。

## 10. Google Drive / rclone

- `rclone copy` または `copyto` を使用する。
- `rclone sync` を使用しない。
- コマンドは引数配列で組み立て、`shell=True` を使用しない。
- 転送元・転送先・終了コードを記録する。
- 認証情報をログに出さない。
- 画像と metadata の両方を検証する。
- 転送失敗は再試行可能な状態として保持する。
- 通常生成とアップスケールの保存先を混在させない。

## 11. Gradio

- `app.py` に全 UI を集中させない。
- タブ・セクション単位で builder を分離する。
- UI handler は入力変換、validation、service 呼び出し、view model 変換に限定する。
- 画像生成を同期 callback 内で完了まで待たない設計を優先する。
- モバイル幅で横スクロールが発生しないことを確認する。
- Advanced settings は Accordion を使用する。
- 主要ボタンは十分な高さと間隔を持たせる。
- Gallery の大量一括読込を避ける。
- 選択中 generation ID を明示的に状態管理する。
- 最近使った設定や最近使った LoRA は別の ViewModel で管理する。

## 12. 履歴・検索

- 履歴検索条件は型付きフィルタとして扱う。
- テキスト検索と構造化検索を分離する。
- 検索結果は paging 前提で取得する。
- 原寸画像は詳細表示時のみ読み込む。
- 一覧はサムネイル優先とする。
- お気に入り、成功/失敗、生成種別は絞り込み可能にする。

## 13. ジョブ・キュー

- 生成、アップスケール、同期はジョブとして扱う。
- キュー順序、状態、再試行可否を保持する。
- UI からキャンセルしても、アプリ側と ComfyUI 側の状態がずれないようにする。
- stale job recovery を考慮する。
- queued / running / completed / failed / cancelled を区別する。

## 14. エラー処理

- ユーザー向けメッセージと詳細ログを分離する。
- `except Exception: pass` を禁止する。
- 例外再送出は `raise ... from exc` を使用する。
- エラーは retryable / non-retryable を区別する。
- checkpoint 不足、LoRA 不足、custom node 不足、GDrive 失敗を別エラー型とする。
- 失敗した generation も監査可能な状態で記録する。

## 15. ログ

- 構造化ログを推奨する。
- generation ID、job ID、prompt ID を context に含める。
- prompt は個人情報を含む可能性があるため、通常ログへ全文を重複出力しない。
- API キー、rclone config、Cookie を出力しない。
- ComfyUI response を無制限にログ出力しない。
- subprocess log は専用ファイルへ保存する。

## 16. テスト

必須ユニットテスト:

- GenerationSettings validation
- LoRA 重複・強度 validation
- workflow template mapping
- metadata serialize / deserialize
- PNG metadata 不在時の sidecar fallback
- generation parent-child relation
- date folder calculation
- safe path resolution
- rclone command construction
- ComfyUI response parsing
- WebSocket 切断後の history recovery
- model catalog filtering
- search filter validation
- preset serialize / deserialize
- storage usage calculation
- upscale limit validation
- prompt diff calculation

統合テスト:

- fake ComfyUI server による prompt 送信
- 生成完了から画像保存まで
- アップスケール親子関係
- GDrive adapter の成功・失敗
- DB migration
- job queue の投入・完了・失敗

通常の CI では GPU、実 ComfyUI、実 Google Drive に接続しない。

## 17. 静的確認

```bash
ruff format --check .
ruff check .
mypy src
pytest
```

UI・ComfyUI・rclone を変更した場合は、手動確認手順も記載する。
