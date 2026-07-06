# nature-downloader

<p align="center">
  <img src="assets/banner.jpg" alt="nature-downloader — 补齐 nature 工作流缺失的 PDF 落地环节" width="100%">
</p>

`nature-downloader` 是一个合法机构权限下的学术全文/PDF 下载 skill。它把两部分能力合在一起：

- **首次资源入口配置**：先记录用户实际使用的图书馆电子资源链接，再识别学校、SSO/CARSI/EZproxy/WebVPN/资源聚合平台信息，配置保存到 `~/.config/lit-dl/school.json`。
- **真实全文落地**：通过用户已经登录的 Chrome 会话和 web-access CDP proxy，复用本人的图书馆/CARSI/出版社访问权限，把能合法访问的 PDF 保存到项目文件夹；如果馆藏只提供 HTML 全文，则保存 HTML/文本并明确说明没有 PDF。
- **中文文献默认知网**：按中文题名下载时，默认优先走用户已登录/已授权的 CNKI/知网页面；可通过学校配置或 `--cnki-url` 指定图书馆提供的知网入口。
- **开放获取优先**：若文献本身是开放获取或开源期刊文章，直接走合法开放 PDF；若图书馆资源明确无权限，直接告诉用户。

它不绕过付费墙，不使用镜像站，不破解验证码，不读取或导出 cookies、密码、localStorage、session 文件。遇到 jAccount/CARSI、验证码、Cloudflare、短信/OTP、人机验证时，会停下让用户本人在 Chrome 里完成。

## 快速使用

把下面这句话复制给 agent，即可让它自动完成首次配置引导：

```text
请使用 nature-downloader 帮我完成首次图书馆资源配置：先向我索要图书馆电子资源/数据库入口链接，识别授权路径并保存配置；如果需要登录，请引导我在 Chrome 中完成统一身份认证/CARSI 登录；最后运行配置展示和连通性自检，确认后续可复用该图书馆资源下载文献。
```

### 1. 配置资源入口

全新用户优先提供图书馆电子资源/数据库入口链接，而不是先填学校名：

```bash
python3 scripts/configure_school.py infer "https://whu.metaersp.cn/personalIndex"
python3 scripts/configure_school.py url "https://whu.metaersp.cn/personalIndex"
python3 scripts/configure_school.py show
```

`infer` 只判断授权路径不保存，`url` 会写入配置。比如武汉大学 `whu.metaersp.cn` 会被识别为资源聚合门户，后续需要登录时再跳转到 `cas.whu.edu.cn`。

如果没有资源入口链接，再用预设学校兜底：

```bash
python3 scripts/configure_school.py preset 上海交通大学
python3 scripts/configure_school.py show
```

查看可用预设：

```bash
python3 scripts/configure_school.py list
```

运行连通性自检：

```bash
python3 scripts/configure_school.py health --force
```

配置路径默认是 `~/.config/lit-dl/school.json`。测试或多 profile 场景可以用 `LIT_DL_CONFIG_DIR=/path/to/configdir` 覆盖。

### 2. 准备浏览器登录态

1. 在 Chrome 里打开本校图书馆或学术资源聚合入口。
2. 用本人账号完成统一身份认证/CARSI 登录。
3. 确认 Web of Science 或目标出版商页面能在 Chrome 中正常看到全文入口。
4. 打开 `chrome://inspect/#remote-debugging`，允许当前浏览器实例远程调试。
5. 启动 web-access CDP proxy。

### 3. 下载文献

按 DOI 下载：

```bash
node scripts/batch_download.mjs \
  --dois "10.1007/s00122-021-03957-1,10.1111/pbi.14066" \
  --out "./文献自动下载"
```

按中文题名下载，默认走知网：

```bash
node scripts/batch_download.mjs \
  --title "乡村振兴背景下数字治理研究" \
  --out "./文献自动下载"
```

如果只要 PDF、不要 CAJ，加 `--cnki-format pdf`。这样在知网没有明确 PDF 下载入口时，脚本会报出未找到可授权 PDF，而不会保存 `.caj`：

```bash
node scripts/batch_download.mjs \
  --title "乡村振兴背景下数字治理研究" \
  --cnki-format pdf \
  --out "./文献自动下载"
```

如果学校图书馆提供了专用知网入口，可显式指定：

```bash
node scripts/batch_download.mjs \
  --title "乡村振兴背景下数字治理研究" \
  --cnki-url "https://kns.cnki.net/kns8s/defaultresult/index" \
  --out "./文献自动下载"
```

按主题从 Web of Science 检索并下载前 N 篇：

```bash
node scripts/batch_download.mjs \
  --topic "rice blast resistance gene" \
  --count 10 \
  --out "./文献自动下载"
```

按精确题名下载开放获取论文（适合 arXiv 论文、无 DOI 论文）：

```bash
node scripts/batch_download.mjs \
  --title "Attention Is All You Need" \
  --open-access \
  --out "./文献自动下载"
```

已知 PDF 地址时直接下载并验证：

```bash
node scripts/batch_download.mjs \
  --pdf-url "https://arxiv.org/pdf/1706.03762" \
  --title "Attention Is All You Need" \
  --out "./文献自动下载"
```

默认只下载主 PDF。只有明确需要补充材料时才加：

```bash
node scripts/batch_download.mjs --dois "10.xxxx/example" --out "./文献自动下载" --si
```

输出目录：

```text
文献自动下载/
  PDFs/
  SupportingInformation/
```

脚本会输出 JSON 状态，常见状态包括 `downloaded`、`open_access_downloaded`、`full_text_html_available`、`library_no_permission`、`carsi_waiting_user`、`publisher_verification_waiting_user`、`sciencedirect_robot_check`、`no_authorized_pdf_found`、`failed_after_retry`。当状态是 `full_text_html_available` 时，表示已拿到可读 HTML 全文，但当前授权路径没有有效 PDF，回复用户时必须说清楚；当状态是 `library_no_permission` 时，表示当前图书馆资源没有该文献全文权限，也必须直接说明。

## 当前实现边界

- 已实现：学校配置、配置文件读写、预设库、连通性自检、Web of Science 入口配置读取、Chrome 登录态下 PDF 下载、PDF 文件头验证、状态码体系。
- 已实现：中文题名默认 CNKI/知网路由，复用 Chrome 中已有的图书馆/知网登录态下载可授权访问的 PDF/CAJ；可用 `--cnki-format pdf` 限制为只下载 PDF。
- 已验证路径：以上海交通大学/SJTU + Web of Science + CARSI/Chrome 会话为主要真实下载路径。
- 新用户首配策略：优先从图书馆资源入口链接识别授权链路；学校预设只作为兜底。
- 已验证开放获取路径：`--title "Attention Is All You Need" --open-access` 会精确匹配 arXiv 标题并下载 PDF。
- 工作流策略：开放获取文章直接下载；非开放获取文章走已配置图书馆资源；馆藏无权限时明确告知用户。
- 可扩展路径：其他学校可通过 `data/schools.yaml` 配置 SSO/CARSI 和 `discovery.web_of_science_url`。
- 不承诺：无登录态下载、绕过出版社限制、自动处理验证码/OTP/Cloudflare、无限批量下载。
- 注意：`--topic` 是 Web of Science 主题检索，不保证精确题名命中；精确题名优先使用 `--title`。

## 依赖

```bash
pip install -r requirements.txt
node --version  # 推荐 Node.js 22+
```

## 验证

```bash
python3 -m unittest discover -s tests/python
node --test tests/unit/*.test.mjs
node --check scripts/batch_download.mjs
node --check scripts/browser_pdf_downloader.mjs
```

详见 `SKILL.md` 了解完整工作流、安全边界、状态码体系和失败处理。
