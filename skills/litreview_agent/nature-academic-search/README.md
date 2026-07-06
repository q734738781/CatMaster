# `nature-academic-search` 技能

`nature-academic-search` 是面向 Claude Code / Codex 工作流的学术检索技能包，集成 CrossRef、PubMed、arXiv、Scopus 和 ScienceDirect 文献数据源。

## 功能

- **多源并发搜索**：默认查询 CrossRef、PubMed、arXiv，并合并返回结果。
- **按 ID 获取详情**：支持 DOI、PMID、arXiv ID 自动识别。
- **格式化引用**：支持 APA、Nature、IEEE、Vancouver 等风格。
- **MeSH 词表查询**：辅助构建精准 PubMed 检索式。
- **文献管理脚本**：支持 `.nbib`、`.ris`、`.bib`、`.enw` 格式互转。
- **Scopus / ScienceDirect**：支持论文、作者、机构、期刊、引用概览、PlumX 指标和 ScienceDirect 元数据检索。
- **严格他引与高影响力引用者审计**：判断引用目标论文的文献是否属于严格他引，排除自引、团队引和明显合作网络引用；可把指定文章的文章名、发表时间、作者名、作者机构、引用数、严格他引数和 DOI 整理成表格；进一步识别院士、校长/院长、杰青/长江、Fellow、高被引学者等高影响力引用者，并提取他们在正文中如何引用目标论文。

## MCP 运行

插件默认通过 `uv` 启动隔离运行环境：

```bash
uv run --no-project --directory <mcp-server> --with "mcp>=1.0.0,<2.0.0" --with "requests>=2.28.0,<3.0.0" --with "toml>=0.10.2,<2.0.0" --with "lxml>=4.9.0,<6.0.0" --with "pybliometrics>=4.4.1,<5.0.0" python academic_search_server.py
```

PubMed 需要在环境变量 `PUBMED_EMAIL` 或 `mcp-server/config.toml` 中配置邮箱。Scopus / ScienceDirect 是可选 provider，复用本机 `pybliometrics` 配置，默认读取 `~/.config/pybliometrics.cfg`；不要把 Elsevier API key 写入插件文件。

`search_papers` 只有在 `sources` 显式包含 `scopus` / `sciencedirect` 时才会调用 Elsevier-backed provider，以避免无意消耗 Elsevier API 配额。

## MCP 工具

| 工具 | 说明 |
|------|------|
| `search_papers` | 默认三源并发搜索；可显式添加 Scopus / ScienceDirect |
| `get_paper_by_id` | 按 DOI、PMID 或 arXiv ID 获取详情 |
| `get_citation` | 生成格式化引用 |
| `lookup_mesh` | 查询 MeSH 词表 |
| `search_scopus` | Scopus 高级检索 |
| `get_scopus_abstract` | Scopus 摘要与详情元数据 |
| `get_scopus_citation_overview` | Scopus 引用概览 |
| `search_scopus_authors` / `get_scopus_author` | 作者检索与详情 |
| `search_scopus_affiliations` / `get_scopus_affiliation` | 机构检索与详情 |
| `search_scopus_serial_titles` / `get_scopus_serial_title` | 期刊与连续出版物检索和详情 |
| `get_scopus_plumx_metrics` | PlumX 指标 |
| `search_sciencedirect` | ScienceDirect 检索 |
| `get_sciencedirect_article_metadata` | ScienceDirect 文章元数据 |

## 严格他引与引用者画像

当用户询问 `严格他引`、`他引判定`、`谁引用了我的文章`、`引用我的文章的人有没有大牛`、`文章引用表`、`指定文章引用数`、`严格他引数`、`整理成表格`、`院士引用`、`杰青引用`、`长江学者引用` 或 `Fellow citation` 时，本技能会加载
`references/workflows/wf6-strict-other-citation-impact-audit.md`。

该 workflow 的默认标准比普通“非自引”更严格：

1. 先确认目标论文的 DOI、作者、机构和可用作者 ID。
2. 汇总 Scopus、Web of Science、Semantic Scholar、OpenAlex、CrossRef 或出版商 cited-by 页面中的 citing papers，并说明覆盖范围。
3. 排除直接自引、同团队/同课题组引用、明显合作网络引用和元数据不足的情况。
4. 如果用户要表格，输出 `文章名 / 发表时间 / 作者名 / 作者机构 / 引用数 / 严格他引数 / DOI / 证据备注`。
5. 对严格他引或可能外部引用者，核验是否存在院士、校长/院长、杰青/长江、Fellow、高被引学者等身份信号。
6. 从可获取全文中抽取 citation context，判断对方是把目标论文作为背景、方法、对比、延伸、复现、批评还是综述性引用。

所有身份标签都必须附带证据来源；无法确认同一人的情况只能标记为 `unverified`。
