# フェーズ1B実装メモ

フェーズ1Bでは、リポジトリで管理する `sdxl_txt2img` API ワークフローを使った、
単一画像の SDXL 生成経路を初めて提供します。

- Gradio 5 のフォームで prompt、サイズ、seed モード、steps、CFG、checkpoint、sampler、scheduler を指定できます。
- ランダム seed `-1` はアプリケーションサービスで一度だけ確定し、確定した seed を結果とともに返します。
- ComfyUI の `/prompt`、WebSocket による進捗、制限付きの `/history/{prompt_id}` 復旧ポーリング、
  `/view` による画像取得はアダプターの責務です。
- 出力画像は Pillow で検証し、Asia/Tokyo の日付フォルダーを使って
  `<data_dir>/generations/YYYY-MM-DD/generated/` 配下へアトミックに保存します。
- このフェーズでは、ジョブと生成結果を意図的にメモリ上だけで管理します。

フェーズ1Bの対象外: LoRA 適用、バッチ生成、VAE 切り替え、SQLite/Alembic による永続化、
PNG metadata、sidecar JSON、履歴検索、プリセット、Google Drive/rclone 同期、アップスケール。

手動確認:

1. 必要な SDXL checkpoint と標準ノードを指定して ComfyUI を起動する。
2. `python -m runpod_sdxl_image_studio.app` を実行する。
3. Gradio のページを開き、システム接続確認を押してから能力情報を再読込する。
4. checkpoint を選択し、prompt を入力して生成を押す。
5. 進捗、生成画像、確定 seed、ローカルの `generated/` 出力ファイルを確認する。
