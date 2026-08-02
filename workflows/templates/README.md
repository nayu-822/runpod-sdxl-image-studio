# ワークフローテンプレート

アプリが利用する固定 ComfyUI ワークフローテンプレートを配置します。任意ワークフローの実行入口にはしません。

Phase 5では `sdxl_image_upscale_api.json` と `sdxl_latent_upscale_api.json` を使用します。テンプレートの選択は
`load_workflow_template()`の固定whitelist経由に限定し、入力画像名、モデル、保存prefix、Latent方式の生成条件だけを
型付きAdapterがバインドします。ユーザーが任意のJSONやnode classを指定することはできません。
