# フェーズ2A実装

フェーズ2Aでは、固定ワークフローによる複数 LoRA と VAE 選択を伴う SDXL 生成の基盤を追加します。

## 実装済み

- `LoraSetting` で相対名、`-2.0` から `2.0` までの強度、0以上の順序、名前の重複、順序値の重複を検証します。
- `GenerationSettings` は `vae_name` と順序付き LoRA 設定タプルを保持します。
- ワークフローアダプターは、選択された LoRA に限って `LoraLoader` ノードを追加し、モデルと CLIP の出力を順番に接続します。
- アダプターは外部 VAE の場合に限って標準の `VAELoader` を追加します。`None` の場合は checkpoint の VAE 出力を使用します。
- 生成前検証では `/prompt` の前に、能力情報への登録、任意ノードの有無、`Settings.max_loras` を確認します。
- Gradio UI には、追加・削除・並べ替え、モデル/CLIP 強度、checkpoint 内蔵 VAE の選択に対応した、
  制限付きでモバイル向けの LoRA 編集画面があります。
- 能力情報の再読込では、引き続き有効な VAE と LoRA の選択を保持し、削除された選択を解除します。

## 手動確認

1. `python -m runpod_sdxl_image_studio.app` でアプリを起動する。
2. ComfyUI へ接続し、能力情報を再読込する。
3. VAE セレクターに `Checkpoint内蔵VAE` と外部 VAE 名が含まれることを確認する。
4. LoRA を2つ追加し、強度を設定して並べ替え、画像を生成する。
5. 結果詳細に VAE と LoRA の順序・強度が表示されることを確認する。
6. 無効または利用できない選択が安全に失敗し、生成ボタンが再び使用可能になることを確認する。

## 後続フェーズへ延期

トリガーワード、カテゴリ、お気に入り、推奨強度、プレビュー、検索、プリセットなどの LoRA metadata は対象外です。
SQLite/Alembic による履歴、バッチ生成、キュー管理、metadata sidecar、Google Drive/rclone、アップスケール、
img2img、RunPod API 連携も延期します。
