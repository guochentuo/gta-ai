#!/bin/bash
# 开发节点 192.168.80.2：把 gta-ai 的日志改动（embedding / OCR）更新到 /opt/gta-ai 并逐个重启、验证；失败自动回退该服务。
# 范围：只动 gta-ai-embedding、gta-ai-ocr 两个单元及其 /opt/gta-ai/<svc>/{deploy/*,app.py,service_logging.py,Containerfile}；
#   新建 /opt/gta-ai/<svc>/deploy.env（只含 GTA_ENV/GTA_NODE/GTA_DEPLOY_SHA，无凭据）；OCR 私有配置只改 OCR_IMAGE 一个键。
#   不动 config 其他键、data、logs、模型、其他单元；旧 OCR 镜像保留；私有配置不入 Git。不触碰生产。
# 前置：已安装文件必须与基线提交逐个一致（防覆盖他人改动），否则停止。
# 用法：deploy-dev-logging.sh <新提交> [基线提交=61345a2]（在本仓库工作区运行）
set -euo pipefail
NEW=$(git rev-parse --verify "${1:?新提交}^{commit}"); BASE=$(git rev-parse --verify "${2:-61345a2}^{commit}")
TAG="dev-${NEW:0:7}"
SSHC=(ssh -F "$HOME/.ssh/gta-servers.conf" room-2)
STAGE=/tmp/gta-ai-stage-${NEW:0:7}
# macOS 自带 bash 3.2 无关联数组，用函数映射
unit_of() { if [[ $1 == paddle-ocr ]]; then echo gta-ai-ocr; else echo gta-ai-embedding; fi; }
health_of() { if [[ $1 == paddle-ocr ]]; then echo http://192.168.80.2:18081/health; else echo http://192.168.80.2:18080/health; fi; }
FILES_ocr=(app.py service_logging.py Containerfile deploy/run-ocr.sh deploy/logrotate.conf)
FILES_emb=(deploy/run-embedding.sh deploy/logrotate.conf)

files_of() { if [[ $1 == paddle-ocr ]]; then echo "${FILES_ocr[@]}"; else echo "${FILES_emb[@]}"; fi; }
remote() { "${SSHC[@]}" "$@"; }

echo "新提交 $NEW  基线 $BASE  OCR 镜像标签 $TAG"
# 1. 前置核对：已安装文件 == 基线
for svc in qwen-embedding paddle-ocr; do
  for f in $(files_of $svc) deploy/$(unit_of $svc).service; do
    git cat-file -e "$BASE:$svc/$f" 2>/dev/null || continue   # 基线中新文件（如 service_logging.py）不检查
    want=$(git show "$BASE:$svc/$f" | shasum -a 256 | cut -c1-64)
    if [[ $f == deploy/*.service ]]; then path=/etc/systemd/system/$(unit_of $svc).service; else path=/opt/gta-ai/$svc/$f; fi
    got=$(remote "sudo -n sha256sum $path 2>/dev/null | cut -c1-64" || true)
    [[ -z "$got" || "$want" == "$got" ]] || { echo "已安装 $path 与基线不同，停止（未修改任何文件）"; exit 1; }
  done
done
OLD_OCR_IMAGE=$(remote "sudo -n grep -h '^OCR_IMAGE=' /opt/gta-ai/paddle-ocr/config/ocr.env /opt/gta-ai/paddle-ocr/config/node.env 2>/dev/null | head -1 | cut -d= -f2")
OCR_IMAGE_FILE=$(remote "sudo -n grep -l '^OCR_IMAGE=' /opt/gta-ai/paddle-ocr/config/ocr.env /opt/gta-ai/paddle-ocr/config/node.env 2>/dev/null | head -1")
[[ -n "$OLD_OCR_IMAGE" && -n "$OCR_IMAGE_FILE" ]] || { echo "找不到 OCR_IMAGE，停止"; exit 1; }
echo "当前 OCR_IMAGE=$OLD_OCR_IMAGE（$OCR_IMAGE_FILE），回退时恢复此值"

# 2. 暂存新提交的两个服务目录（只用于构建与复制）
git archive "$NEW" qwen-embedding paddle-ocr | remote "rm -rf $STAGE && mkdir -p $STAGE && tar -xf - -C $STAGE"

install_files() {  # install_files <svc> <源目录或 git:<commit>>
  local svc=$1 src=$2
  for f in $(files_of $svc); do
    if [[ $src == git:* ]]; then
      git cat-file -e "${src#git:}:$svc/$f" 2>/dev/null || { remote "sudo -n rm -f /opt/gta-ai/$svc/$f"; continue; }
      git show "${src#git:}:$svc/$f" | remote "sudo -n install -D -o root -g root -m 0644 /dev/stdin /opt/gta-ai/$svc/$f"
    else
      remote "sudo -n install -D -o root -g root -m 0644 $src/$svc/$f /opt/gta-ai/$svc/$f"
    fi
  done
  remote "sudo -n chmod 0755 /opt/gta-ai/$svc/deploy/*.sh"
  local unit; unit=$(unit_of $svc)
  if [[ $src == git:* ]]; then git show "${src#git:}:$svc/deploy/$unit.service" | remote "sudo -n install -o root -g root -m 0644 /dev/stdin /etc/systemd/system/$unit.service"
  else remote "sudo -n install -o root -g root -m 0644 $src/$svc/deploy/$unit.service /etc/systemd/system/$unit.service"; fi
  remote "sudo -n systemctl daemon-reload"
}

wait_health() {  # wait_health <svc>：最多 300 秒
  for _ in $(seq 1 60); do remote "curl -fsS -o /dev/null -m 5 $(health_of $1)" && return 0; sleep 5; done; return 1
}

rollback() {  # rollback <svc>
  local svc=$1; echo "!! $svc 验证失败，回退到基线 $BASE"
  install_files $svc git:$BASE
  remote "sudo -n rm -f /opt/gta-ai/$svc/deploy.env"
  [[ $svc == paddle-ocr ]] && remote "sudo -n sed -i 's#^OCR_IMAGE=.*#OCR_IMAGE=$OLD_OCR_IMAGE#' $OCR_IMAGE_FILE"
  remote "sudo -n systemctl restart $(unit_of $svc)"
  wait_health $svc && echo "$svc 已回退并恢复健康" || echo "$svc 回退后仍不健康，需要人工"
  exit 1
}

deploy_env() {
  printf 'GTA_ENV=dev\nGTA_NODE=r740xd-2\nGTA_DEPLOY_SHA=%s\n' "$NEW" | remote "sudo -n install -o root -g root -m 0644 /dev/stdin /opt/gta-ai/$1/deploy.env"
}

# 3. embedding：脚本/单元 + deploy.env，重启，验证健康与 JSON 日志
install_files qwen-embedding $STAGE; deploy_env qwen-embedding
remote "sudo -n systemctl restart gta-ai-embedding"
wait_health qwen-embedding || rollback qwen-embedding
echo "embedding 健康"

# 4. OCR：先构建新镜像（旧镜像保留），再换文件与镜像标签，重启验证
remote "cd $STAGE/paddle-ocr && sudo -n podman build -q -t localhost/gta-ai-ocr:$TAG -f Containerfile ." || { echo "OCR 镜像构建失败，OCR 未改动"; exit 1; }
install_files paddle-ocr $STAGE; deploy_env paddle-ocr
remote "sudo -n sed -i 's#^OCR_IMAGE=.*#OCR_IMAGE=localhost/gta-ai-ocr:$TAG#' $OCR_IMAGE_FILE"
remote "sudo -n systemctl restart gta-ai-ocr"
wait_health paddle-ocr || rollback paddle-ocr
echo "OCR 健康（镜像 localhost/gta-ai-ocr:$TAG；旧镜像 $OLD_OCR_IMAGE 保留）"
remote "rm -rf $STAGE"
echo "完成。回退：恢复 OCR_IMAGE=$OLD_OCR_IMAGE，并以基线 $BASE 的文件重装两个服务后重启"
