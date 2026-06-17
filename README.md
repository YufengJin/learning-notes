# Learning Notes

个人学习笔记知识库，**MkDocs + Material** 构建，发布在
**<https://yufengjin.github.io/learning-notes/>**。主题与
[paper-snapshots](https://github.com/YufengJin/paper-snapshots) 统一：Claude 暖灰底
(`#F8F8F6`) 配橙色 (`#FB5D00`)，极简浅色风格（见 `docs/stylesheets/extra.css`）。

## 添加笔记

```bash
# 1) 在某个分区下新建 markdown
$EDITOR docs/ml/my-note.md
# 2) 在 mkdocs.yml 的 nav 里登记该页
# 3) 提交推送 —— GitHub Actions 自动构建并部署到 gh-pages 分支
git add . && git commit -m "notes: add my-note" && git push
```

## 本地预览（可选，需要 Python）

```bash
pip install -r requirements.txt
mkdocs serve            # http://127.0.0.1:8000
```

## 结构

```
mkdocs.yml                 # 配置（主题 / 导航 / 扩展 / 插件）
docs/
  index.md                 # 首页
  ml/ math/ programming/ reading/   # 分区（每区一个 index.md 落地页）
  stylesheets/extra.css    # 统一主题：暖灰 + 橙
  javascripts/mathjax.js   # 公式渲染
.github/workflows/deploy.yml   # push 到 main 即自动部署
```

## 部署机制

push 到 `main` → GitHub Actions 跑 `mkdocs gh-deploy` → 构建产物推到 `gh-pages` 分支 →
GitHub Pages 从 `gh-pages` 分支发布。**只需在仓库 Settings → Pages 把 Source 设为 `gh-pages` 分支**（首次已配置）。
