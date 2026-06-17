#!/usr/bin/env bash
# 本地预览 learning-notes（带子路径 /learning-notes/）
# 用法: ./serve.sh   然后浏览器打开下面打印的地址
set -euo pipefail
cd "$(dirname "$0")"

echo "👉 打开: http://127.0.0.1:8000/learning-notes/"

if command -v mkdocs >/dev/null 2>&1; then
  exec mkdocs serve
elif command -v uv >/dev/null 2>&1; then
  # 临时环境，不污染全局；改 md 自动热重载
  exec uv run --with mkdocs-material mkdocs serve
else
  echo "需要 mkdocs 或 uv。安装其一：" >&2
  echo "  pip install -r requirements.txt   # 然后 mkdocs serve" >&2
  exit 1
fi
