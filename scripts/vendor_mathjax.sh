#!/bin/sh
# MathJax 自托管的来源与更新脚本（供应链可追溯）。
# 当前版本：mathjax@3.2.2（Apache-2.0，https://github.com/mathjax/MathJax）
# 来源 CDN：https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/
# 升级：改 VER 重跑本脚本，然后 mkdocs build --strict 验证公式页。
set -eu
VER="3.2.2"
BASE="https://cdn.jsdelivr.net/npm/mathjax@${VER}/es5"
DEST="$(dirname "$0")/../docs/javascripts/vendor/mathjax"

mkdir -p "$DEST/output/chtml/fonts/woff-v2"
curl -fsSL -o "$DEST/tex-mml-chtml.js" "$BASE/tex-mml-chtml.js"
# Apache-2.0 §4：再分发必须附许可证副本
curl -fsSL -o "$DEST/LICENSE" "https://cdn.jsdelivr.net/npm/mathjax@${VER}/LICENSE"
for f in MathJax_AMS-Regular MathJax_Calligraphic-Bold MathJax_Calligraphic-Regular \
         MathJax_Fraktur-Bold MathJax_Fraktur-Regular MathJax_Main-Bold \
         MathJax_Main-Italic MathJax_Main-Regular MathJax_Math-BoldItalic \
         MathJax_Math-Italic MathJax_Math-Regular MathJax_SansSerif-Bold \
         MathJax_SansSerif-Italic MathJax_SansSerif-Regular MathJax_Script-Regular \
         MathJax_Size1-Regular MathJax_Size2-Regular MathJax_Size3-Regular \
         MathJax_Size4-Regular MathJax_Typewriter-Regular MathJax_Vector-Bold \
         MathJax_Vector-Regular MathJax_Zero; do
  curl -fsSL -o "$DEST/output/chtml/fonts/woff-v2/$f.woff" \
    "$BASE/output/chtml/fonts/woff-v2/$f.woff"
done
echo "vendored mathjax@${VER} -> $DEST"
