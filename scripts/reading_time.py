#!/usr/bin/env python3
"""Byline 阅读时长的唯一口径（手算必然漂移，交稿前跑一遍照抄）。

    python3 scripts/reading_time.py docs/*/*.md

规则（与 CLAUDE.md 的 Byline 约定一致）：
- 中文字数 ÷ 350 + 英文单词数 ÷ 200，四舍五入（half-up，非银行家舍入），至少 1 分钟。
- 统计前剔除：YAML front-matter、fenced code、行内 code、HTML 标签、
  $...$ / $$...$$ 数学、URL。公式/代码/交互 demo 的阅读成本不折算进时长，
  所以对公式密集的笔记这是**下界估计**——byline 写「约 N 分钟」而非精确承诺，口径以此脚本为准。
"""
import math
import re
import sys


def minutes(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)          # front-matter
    text = re.sub(r"```.*?```", " ", text, flags=re.S)                  # fenced code
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)                # display math
    text = re.sub(r"\$[^$\n]+\$", " ", text)                            # inline math
    text = re.sub(r"`[^`\n]+`", " ", text)                              # inline code
    text = re.sub(r"<[^>]+>", " ", text)                                # html tags
    text = re.sub(r"https?://\S+", " ", text)                           # urls
    cjk = len(re.findall(r"[㐀-鿿]", text))
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))
    return cjk, words, max(1, math.floor(cjk / 350 + words / 200 + 0.5))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p.endswith("/index.md") or "/style/" in p:
            continue
        cjk, words, m = minutes(p)
        unit = "min read" if p.endswith(".en.md") else "分钟"
        print(f"{p}: {cjk} 汉字 + {words} 词 → 约 {m} {unit}")
