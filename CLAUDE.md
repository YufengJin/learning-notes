# learning-notes — project notes

个人学习笔记知识库，**MkDocs + Material** 构建，发布在
**`https://yufengjin.github.io/learning-notes/`**（仓库 `git@github.com:YufengJin/learning-notes.git`，分支 `main`）。
个人主页 `yufengjin.github.io` 页脚有入口链接。

## 布局
- `mkdocs.yml` — 配置（主题 / 顶部 tab + 左侧栏 / 扩展 / 插件 / nav）。
- `docs/` — 内容：`index.md`（首页）+ 分区 `ml/ robotics/ math/ reading/`（每区一个 `index.md` 落地页）。
- `docs/style/index.md` — **样式模板页**，枚举所有常用格式；**不进顶部 tab**，仅从首页底部链接（`mkdocs.yml` 里 `validation.nav.omitted_files: ignore` 已放行）。
- `docs/stylesheets/extra.css` — 统一主题（见下）。`docs/javascripts/mathjax.js` — 公式渲染。
- `.github/workflows/deploy.yml` — push 到 `main` 即 `mkdocs gh-deploy` 自动构建并部署到 `gh-pages` 分支。
- `requirements.txt` — `mkdocs-material` + `mkdocs-static-i18n`（双语，见下）。

## 统一风格（与 paper-snapshots 一致，务必保持）
Claude 暖灰底 + 石墨灰强调，极简近单色，**单一浅色主题**（不加深色切换、不引入橙色或其它强调色）。
全部在 `docs/stylesheets/extra.css` 的设计 token 里：

| token | 值 | token | 值 |
|---|---|---|---|
| `--ln-bg` | `#F8F8F6` | `--ln-ink` | `#201F1C` |
| `--ln-surface` | `#FFFFFF` | `--ln-ink-2` | `#3A3833` |
| `--ln-surface-2` | `#F1EFEA` | `--ln-muted` | `#6B675F` |
| `--ln-line` | `#E8E4DC` | `--ln-acc`（石墨灰） | `#5A5953` |

- 头部/顶部 tab 用浅色（不要彩色横幅），石墨灰只点缀链接 / 激活态 / 重点。
- 字体：正文 Inter + 中文系统字体回退；代码 JetBrains Mono。
- 调样式只改 `extra.css`，并同步参照 `docs/style/` 页确认观感。

## 双语
`mkdocs-static-i18n` 提供中英双语，**语言选择器是 Material 原生的，在右上角 header**
（插件自动注入 `extra.alternate`，不要手写切换器）。

- 中文在根路径 `/learning-notes/`，英文在 `/learning-notes/en/`。
- `docs_structure: suffix` —— 英文版是同名 `.en.md`（如 `docs/ml/index.en.md`）。
- **口径（2026-08-05 起）：每篇正文笔记必配 `.en.md`**，与 paper-snapshots「正文双语」对齐。
  `docs/*/index.md` 这类分区首页豁免。英文版是中文稿的忠实全译——结构逐节对应、不增删章节、
  数字与公式逐字一致。
- `fallback_to_default: true` —— 没写 `.en.md` 的页面在 `/en/` 下自动显示中文原文。
  这是**兜底而非许可**：漏写不会坏站、`--strict` 也不报错，所以只能靠自查。交付前跑：
  ```bash
  for f in docs/*/*.md; do case "$f" in *.en.md|*/index.md) continue;; esac; \
    [ -f "${f%.md}.en.md" ] || echo "MISSING EN: $f"; done
  ```
  当前基线：5 篇正文笔记全部有 `.en.md`，该命令无输出。
- nav 标题的英文在 `mkdocs.yml` 的 `nav_translations` 里；新增分区记得同步加一条。
- 插件声明在 `search` 之后（它会接管并按语言重配 search）。改动 plugins 顺序后
  务必验证中文搜索仍可用——CJK separator 容易被插件覆盖掉。
- `theme.features` 不能有 `navigation.instant`：它与 Material 的语言选择器（contextual
  link）不兼容，`mkdocs-static-i18n` 会在构建时报 WARNING，`--strict` 下直接中断构建。

## 写作规范（learning-note skill）
教学笔记的写作与移植规则固化在 **`.claude/skills/learning-note/SKILL.md`**（已入库）：
材料忠实渲染（数字逐字回查、三级论断标签、范围词对齐证据面）、结构契约
（导语/引言/动机/预备知识三元组/正文四拍/图注/References）、分类判据
（**以读者会去哪里找它为准**）、交互 demo 规则（浅色、命名空间前缀、
document$.subscribe + 存在性守卫）、注册三件套（nav + nav_translations +
extra_javascript）。把材料/草稿/现成 blog 变成笔记时触发该 skill。

## 添加笔记
```bash
$EDITOR docs/<分区>/my-note.md     # 1) 新建 markdown
# 2) 在 mkdocs.yml 的 nav 里登记该页（否则不会出现在导航）
git add . && git commit -m "notes: add my-note" && git push   # 3) Action 自动部署
```
- 本地预览：`pip install -r requirements.txt && mkdocs serve`（带子路径 `http://127.0.0.1:8000/learning-notes/`）。
- 写作直接用样式页里的格式：admonition 记录框 / 代码高亮 / 标签页 / 表格 / `$...$` 数学。
- **中文稿 + 同名 `.en.md` 英文全译，两份都要**（分区 index 页豁免）。漏写只会静默回退中文、
  不报错，所以用上面「双语」小节的核对命令自查。

## 资产规则（保持仓库精简）
- **图片一律先压缩再入库**：转 WebP `cwebp -q 80-85 -resize 1280-1600 0`，放在 `docs/<分区>/img/` 下相对引用。
- **不提交原始大图 / PDF / 大视频**；大媒体走外部托管或图床，不进 git 历史。
- 单文件远低于 ~9MB；上限是 GitHub Pages **1GB 构建站点**。

## 写作规范
- **解释、叙述用中文**；但**学术 / 技术术语保留英文原文**，不要硬翻译。
  - 例：`tokenization`、`mode averaging`、`DCT`、`energy compaction`、`Prefix-LM`、`straight-through estimator`、`cross-entropy`、`flow matching` 等保持英文；周围的说明文字用中文。
  - 首次出现可中英并列（如「能量压缩（energy compaction）」），其后用英文原词即可。
  - 公式、变量、代码、库名/类名一律英文原文。

## 命名规范
- 目录 / 文件名一律 **小写 kebab-case**（`a-z 0-9 -`）：不用空格、大写、下划线或中文（URL 友好）。
- 笔记 md 如 `linear-regression.md`；分区目录 ascii 小写（`ml/ robotics/ math/ reading/`）。
- 图片描述性 kebab-case，放 `docs/<分区>/img/`，扩展名小写 `.webp`。
- 中文只用于 `nav` 标题与页面正文，**不进路径/文件名**。

## 其它约定
- **不要手改 `gh-pages` 分支**（它是 Action 的构建产物）。
- 不提交构建产物 `site/`、`.venv/`、`.cache/`、`.DS_Store`（已在 `.gitignore`）。
- 内部链接用相对路径；指向 paper-snapshots 的跨站链接用同域绝对路径 `/paper-snapshots/`。
- nav 改动后跑一次 `mkdocs build --strict` 自查断链。
