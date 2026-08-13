# RunPod本番デプロイ

このdirectoryは、既存のRunPod ComfyUI applicationとImage Studio runtimeを
Phase 13でFresh Podへ配置する手順を説明します。RunPod APIによるPod自動Deployや
外部orchestrationは実装しません。Phase 12のself-only Safe Auto-Terminateは、
application側の既存機能として扱います。

## イメージの内容と境界

imageは固定したbase image
`runpod/comfyui:1.4.4-cuda12.8`からbuildします。repository管理下のImage Studio
sourceとPython package metadata、Alembic migration、workflow定義、
`/opt/image-studio-venv`のPython 3.11以上のvirtual environment、checksum検証済み
rclone 1.74.2、deployment scriptを含みます。

checkpoint、LoRA、VAE、upscaler、生成画像、SQLite state、`rclone.conf`、OAuth
token、API key、cookie、`.env`はimageへ含めません。modelとstateは起動後に既存の
Phase 11/10 application serviceが復元します。

applicationは`/opt/image-studio`へeditable installします。これによりrepository root、
`alembic.ini`、`alembic/`、`workflows/`を既存startup codeから利用できます。
Image StudioのvenvはComfyUIのvenvと分離します。

## ビルドと公開

Dockerが利用できる環境でbuildし、追跡可能なtagをpublishします。production Template
では`latest`をpublishまたは設定しません。

~~~bash
git rev-parse --short HEAD
docker build --platform linux/amd64 -t runpod-sdxl-image-studio:phase13-<git-short-sha> .
docker tag runpod-sdxl-image-studio:phase13-<git-short-sha> <registry-user>/runpod-sdxl-image-studio:phase13-<git-short-sha>
docker push <registry-user>/runpod-sdxl-image-studio:phase13-<git-short-sha>
~~~

build時にdownloadするのは固定したrclone releaseとchecksum fileだけです。
runtimeのbootstrapは`git clone`、`pip install`、`apt`、`curl | bash`、model downloadを
実行しません。

## rclone Secret

localの`rclone.conf`をbase64化し、RunPod Secret
`image_studio_rclone_config_b64`へ登録します。値をcommitせず、Docker build argumentにも
入れません。runtimeではparent directoryを0700、decoded fileを0600として
`/run/image-studio/rclone.conf`へ作成します。

PowerShell:

~~~powershell
$bytes = [IO.File]::ReadAllBytes('.\rclone.conf')
$secret = [Convert]::ToBase64String($bytes)
Set-Clipboard $secret
~~~

Linux/macOS:

~~~bash
RCLONE_CONFIG_B64="$(base64 -w0 ./rclone.conf)"
echo "base64 value prepared"
unset RCLONE_CONFIG_B64
~~~

値はRunPod Secretsにだけ貼り付けます。Template environmentには次を設定します。

~~~dotenv
IMAGE_STUDIO_RCLONE_CONFIG_B64={{ RUNPOD_SECRET_image_studio_rclone_config_b64 }}
RCLONE_CONFIG=/run/image-studio/rclone.conf
RCLONE_REMOTE=drive
~~~

bootstrapは空値、invalid value、設定pathまたはparentのsymlink、decode/write failureを
rejectします。temporary fileへ書いて0600を設定し、atomic replaceします。その後、
`RCLONE_REMOTE`が`rclone listremotes`に存在することだけを確認します。config本文は
表示もlog出力もしません。state sync、remote model preparation、configured remoteの
いずれかを有効にして有効なconfigがない場合はstartupをfail-fastします。

## Template設定

`template.env.example`をRunPod Template environmentへコピーし、Secret placeholderだけを
RunPod Secret機構で置き換えます。DockerのEntrypointは、イメージ側のDockerfileで
`ENTRYPOINT []`としてbase imageの`/start.sh`を解除します。これはTemplateの設定を
変更するという意味ではありません。Template側ではEntrypointとStart Commandを上書きせず、
イメージのCMDでbootstrapを起動し、bootstrapが`/start.sh`をbackground起動します。
production Templateの推奨値は次のとおりです。

| 設定 | 値 |
| --- | --- |
| Name | SDXL Image Studio |
| Category | NVIDIA |
| Container image | 固定したphase13-<git-short-sha> tag |
| Container disk | 50 GB |
| Volume disk | 0 GB |
| Network volume | none |
| HTTP port | 7860/http only |
| Public Template | off |
| Docker Entrypoint | unchanged; Templateで上書きしない（image側は`ENTRYPOINT []`） |
| Docker Start Command | unchanged; image CMDがbootstrapを起動 |

productionでは8188/http、8080/http、8888/httpを公開しません。ComfyUIにはImage
Studioから`127.0.0.1:8188`で接続します。debug時だけ一時的に22/tcpを追加できますが、
production Templateには含めません。RunPodが提供する`RUNPOD_POD_ID`と`RUNPOD_API_KEY`を
Template fileへ追加・保存しません。

## 起動フロー

bootstrapの順序は次のとおりです。

1. optionalなrclone configをmaterializeして検証する。
2. base image既存の`/start.sh`をbackgroundで起動する。
3. 固定endpoint `http://127.0.0.1:8188/system_stats`がreadyになるまで待つ。
   `IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS`はwall-clock deadlineで計測し、
   既定900秒を超えて待ちません。timeout=0では即時probeを最大1回だけ行います。
4. `/opt/image-studio-venv/bin/runpod-sdxl-image-studio`を
   `/opt/image-studio`から起動する。
5. Image Studio起動後、各probe完了後に既定5秒を目安として同じComfyUI endpointを
   再確認する。成功時はfailure countを0へ戻し、
   `IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD`（既定12）回連続失敗した場合だけ
   persistent failureとする。失敗したprobeのHTTP timeoutもあるため、厳密に5秒ごととは
   限定しない。
6. persistentなComfyUI failureまたはchild processの予期しない終了時は、Image Studio、
   base processの順にgraceful TERMを送り、設定されたgrace period後に必要ならKILLへ
   fallbackしてnon-zero終了する。ComfyUIのauto-restartや`sleep infinity`は行わない。
7. SIGTERM/SIGINTではImage Studioとbase processへgraceful TERMを送り、
   `IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS`までbounded waitする。

Docker HEALTHCHECKはuser-facingな`http://127.0.0.1:7860/`だけをprobeし、Fresh Pod用の
startup grace periodは30分です。8188をhealth markerとして公開・追加しません。
healthcheck failureはSQLiteを変更せず、RunPod DELETEも発行しません。bootstrap logは
安全なstatusだけを出し、rclone config本文、token、API key、prompt、raw HTTP responseを
出力しません。

bootstrapはstate restore、Alembic migration、model download、`/prompt`、Drive sync、
Auto-Terminateを実行しません。既存application startup pathがrestore検証とAlembic
upgradeを担当し、Phase 11のmodel-transfer workerが必要modelを準備し、Phase 12の
applicationがreadiness確認後にSafe Auto-Terminateを担当します。

## ローカルとコンテナの確認

GPU不要のsmoke testはPython、package import、Gradio major version、rclone availability、
repository file、executable script、一時SQLiteの`upgrade_database`とintegrity checkを
検証します。GPU、ComfyUI、RunPod、Google Drive、実rclone remoteは必要ありません。

~~~bash
bash -n deploy/runpod/bootstrap.sh
bash -n deploy/runpod/smoke-test.sh
pytest tests/unit/test_phase13_runpod_deployment.py
~~~

image build成功後は次を実行します。

~~~bash
docker run --rm --entrypoint /bin/bash runpod-sdxl-image-studio:phase13-<git-short-sha> /opt/image-studio/deploy/runpod/smoke-test.sh
~~~

Dockerが利用できない場合、Docker buildとcontainer smoke testは未実施と報告します。
static checkやlocal testだけではimage buildや実RunPod Templateの動作を証明できません。

## Fresh Pod確認

Fresh Pod Aは`IMAGE_STUDIO_AUTO_TERMINATE_ENABLED=false`でdeployし、次を確認します。

1. `/start.sh`がbase serviceを起動し、ComfyUIがlocal readinessへ到達する。
2. Image StudioがRunPod proxyの7860から到達できる。
3. rclone remote検証がconfig本文を出さずに成功する。
4. Google Drive state restoreが完了するか、安全なmessageでfail-closedする。
5. 前回form stateが復元され、必要modelが正確に準備される。
6. startup `/prompt`が送信されない。
7. 利用者のGenerateが成功する。
8. imageとmetadataがSYNCEDになり、manifestとfinal state backupがcleanになる。

同じ固定Template tagでFresh Pod Bをdeployし、state、form settings、必要modelが新しい
startup generationなしに復元されることを確認します。AとBが成功してから、別の検証Podで
Auto-Terminateを有効化します。

~~~dotenv
IMAGE_STUDIO_AUTO_TERMINATE_ENABLED=true
~~~

Generation completion、Drive/manifest synchronization、final state backup、grace period、
SAFE TO TERMINATE、self-only DELETE一回、ambiguous response時のPhase 12 fail-closedを
確認します。実Pod確認はCIやlocal Mock/Fake testとは別です。

## トラブルシューティング

- ComfyUI timeout: base image logを確認し、固定local endpointが利用できることを確認します。
  外部URLやuser-supplied URLへ置き換えません。
- rclone config failure: local configからSecretを作り直し、Secret名と`drive:` remoteを
  確認します。`rclone.conf`を表示せず、diagnosticで`rclone config show`を使いません。
- model unavailable: applicationのfail-closed状態を維持し、Phase 11 model preparation UIを
  使用します。modelをimageへコピーしたり、別checkpointへ自動代替したりしません。
- state restore failure: 既存のfail-closed write protectionを維持し、許可されたapplication
  logだけを確認します。bootstrapは`latest.json`のdownloadやSQLite reconciliationをしません。
- container exit: supervised processの終了理由をsafe bootstrap messageで確認します。
  bootstrapはunexpected exitをinfinite sleepで隠しません。
