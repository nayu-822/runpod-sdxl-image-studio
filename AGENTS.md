# AGENTS.md

## プロジェクト概要

このリポジトリは、RunPod GPU Pod 上で ComfyUI をバックエンドとして稼働する、自分専用の Gradio アプリケーション `RunPod SDXL Image Studio` を実装する。

本ツールは、SDXL 画像生成、checkpoint 選択、複数 LoRA 適用、生成履歴検索、生成条件の復元、プリセット、画像アップスケール、Google Drive 保存、生成キュー、システム状態表示、およびスマートフォン向け UI に対応する。

変更を行う前に、次の文書を順番に確認すること。

1. `RUNPOD_SDXL_IMAGE_STUDIO_DESIGN_SPEC.md`
2. `DEVELOPMENT_PLAN.md`
3. `CODING_RULES.md`

文書間で指示が矛盾する場合は、次の優先順位に従うこと。

1. ユーザーから直近で明示された指示
2. `AGENTS.md`
3. `CODING_RULES.md`
4. `DEVELOPMENT_PLAN.md`
5. 設計仕様書

## 基本構成

- UI: Gradio Blocks
- 言語: Python 3.11 以上
- 実行環境: RunPod GPU Pod
- 画像生成バックエンド: ComfyUI
- ComfyUI 連携: HTTP API および WebSocket
- 永続化: SQLite および JSON sidecar
- クラウド保存: rclone 経由の Google Drive
- 外部アクセス: RunPod HTTP Proxy
- コード配布: Git リポジトリ

GPU を使用する画像生成・アップスケール処理を、ユーザーのローカル PC やスマートフォンへ移さないこと。

## アーキテクチャ上の必須ルール

- Gradio UI から ComfyUI API を直接呼ばないこと。
- UI、Application Service、Domain、Adapter、Persistence を分離すること。
- ComfyUI 固有の workflow JSON 生成・変換は Adapter 層へ閉じ込めること。
- 画像生成設定は型付きモデルとして扱うこと。
- ComfyUI の node ID を UI やドメインロジックへ露出させないこと。
- workflow template の schema version を保持すること。
- 画像生成結果は ComfyUI の一時履歴だけに依存しないこと。
- ブラウザ切断後も生成処理を継続し、再接続後に状態を復元できること。
- 生成、アップスケール、同期はジョブまたはキュー管理の対象とすること。

## 再現性

画像ごとに最低限以下を保存すること。

- generation ID
- 作成日時
- positive prompt
- negative prompt
- seed
- width / height
- steps
- CFG
- sampler
- scheduler
- checkpoint 名
- checkpoint の識別情報または hash
- VAE
- LoRA 名、model strength、clip strength、順序
- workflow template ID / version
- ComfyUI prompt ID
- アプリ版
- ComfyUI 版
- 生成画像 SHA-256
- 親画像 ID
- 通常生成 / アップスケール / 派生生成の種別
- アップスケール方式、倍率、denoise、モデル
- Google Drive 保存先
- 同期状態
- お気に入り、メモ、タグ

同条件再生成は「現在の UI 値」ではなく、保存された generation snapshot から実行すること。

## メタデータ

- PNG metadata と sidecar JSON の両方を保存すること。
- PNG metadata を唯一の正としないこと。
- sidecar JSON には `schema_version` を持たせること。
- 外部から読み込んだ画像の metadata は信頼できない入力として検証すること。
- 未知のフィールドは原則保持してもよいが、実行パラメータとして暗黙使用しないこと。
- checkpoint や LoRA が見つからない場合、勝手に代替せず UI へ不足項目を表示すること。

## 画像保護

- 生成済みの元画像をアップスケール時に上書きしないこと。
- アップスケール画像は新しい generation として保存すること。
- 親子関係を DB に保存すること。
- 画像ファイルの確定前に一時ファイルへ保存し、検証後に atomic replace すること。
- 同名ファイルで上書きしないこと。
- 削除操作は初期フェーズでは実装しないか、復元可能な論理削除とすること。

## ComfyUI 連携

- `/prompt` へ送信した prompt ID を保存すること。
- WebSocket から進捗と完了通知を受信すること。
- WebSocket 切断時は `/history/{prompt_id}` による復旧確認を可能にすること。
- ComfyUI の出力ファイルを検証してからアプリ管理領域へコピーすること。
- workflow JSON は許可されたテンプレートから組み立てること。
- UI から任意の node class や Python コードを注入できないようにすること。
- カスタムノード不足は起動時または生成前検証で検出すること。

## モデル・LoRA 管理

- checkpoint、VAE、LoRA、upscaler は許可されたディレクトリ配下だけを列挙すること。
- シンボリックリンクを考慮し、許可ディレクトリ外への脱出を防止すること。
- モデル名は相対パスとして保存し、絶対パスをユーザー向け metadata に露出しないこと。
- 一覧更新操作を用意すること。
- LoRA は複数選択できること。
- 同じ LoRA の重複指定を禁止すること。
- LoRA 強度の範囲を設定モデルで検証すること。
- LoRA にトリガーワード、推奨強度、推奨モデル、カテゴリ、お気に入り、プレビュー画像を持てるようにすること。
- checkpoint と LoRA の互換性は完全自動判定できないため、警告と失敗理由を明確にすること。

## Google Drive

- 通常保存には `rclone copy` または `copyto` を使用すること。
- `rclone sync` は使用しないこと。
- 通常生成とアップスケールを別フォルダに保存すること。
- 日付はユーザーのタイムゾーン（Asia/Tokyo）で決定すること。
- 画像と JSON の両方が保存できた時点で同期成功とすること。
- 同期失敗時も RunPod 内の画像を保持すること。
- 同期は再試行可能であること。
- rclone の認証情報を UI、ログ、DB に保存しないこと。
- 同期状態は pending / syncing / synced / failed を持つこと。

## スマートフォン UI

- 主要操作を横スクロールなしで使用できること。
- 生成ボタンは sticky action または常に押しやすい位置に配置すること。
- 高度な項目は Accordion に格納すること。
- Prompt 欄は十分な高さを確保すること。
- LoRA は行追加式またはカード式とし、各行で名前と強度を編集できること。
- 生成画像は 1 カラムを基本とすること。
- タップ領域を小さくしすぎないこと。
- 大量の履歴を一度に表示しないこと。
- 画像選択状態を見た目で明確にすること。
- 最近使った設定、最近使ったモデル、最近使った LoRA を素早く呼び出せること。
- UI の一時状態だけを正としないこと。

## 長時間処理

画像生成、アップスケール、Google Drive 転送などはジョブとして管理すること。

最低限以下を記録すること。

- Job ID
- Generation ID
- ComfyUI prompt ID
- PID（外部プロセス使用時）
- 状態
- 進捗
- 開始日時
- 終了日時
- エラー概要
- 再試行可否
- ログパス
- キュー順序

## システム状態

システム状態画面または相当機能で以下を確認可能にすること。

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

## セキュリティ

- `.env`、API キー、アクセストークン、Cookie、rclone 設定を Git へ含めないこと。
- ComfyUI API を RunPod HTTP Proxy 外へ直接公開しないこと。
- 任意 URL からのモデルダウンロードは初期スコープに含めないこと。
- ファイル名、画像 metadata、workflow JSON、モデルパスは信頼できない入力として扱うこと。
- ディレクトリトラバーサル、SSRF、任意コマンド実行を防止すること。
- 秘密情報をログへ出力しないこと。

## 実装対応後に必ず報告する内容

1. 実装・変更内容の概要
2. 変更したファイル一覧
3. 重要な実装上・設計上の判断
4. 実行したテストおよび確認コマンド
5. テストおよび確認結果
6. 未解決事項、既知の問題、残っているリスク
7. 手動確認手順
8. 推奨する Git コミットメッセージ

ユーザーから明示的な依頼がない限り、`git commit`、`git push`、破壊的な Git 操作を実行しないこと。
