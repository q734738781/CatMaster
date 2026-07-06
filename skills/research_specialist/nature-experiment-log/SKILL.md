---
name: nature-experiment-log
description: "标准化实验日志记录——接收原始材料（图/语音/文字），产出带 YAML frontmatter 的标准日志到 Obsidian vault。需配合飞书 CLI 或手动输入使用。"
version: 1.0.0
author: 十五 (JL Lab)
license: MIT
metadata:
  hermes:
    tags: [research, experiment, logging, feishu, obsidian, automation]
    related_skills: [nature-literature-pipeline, feishu-cli-integration]
---

# experiment-log — 实验日志标准化

## 触发条件

用户通过以下任一方式提交实验原始材料时自动加载：

- **路径 A** — CLI 直接提交（图片 / 语音转录 / 文字）
- **路径 B** — 发到飞书科研群，通过 `feishu-cli-integration` 扫描

## 前置依赖

需要配合 `feishu-cli-integration` skill 使用。确保：
- 飞书 bot 已添加到目标群
- bot 权限：`im:message` + `im:resource` + `im:message.group_msg`
- 群 ID 配置在 skill 上下文中

## 处理流程

1. 接收材料 → vision_analyze 读图 + 提取结构化信息
2. 生成实验 ID + 样品批次 ID
3. 写出标准日志到 `wiki/实验日志/{体系}/{类型}/{exp_id}.md`
4. 原始材料（图片等）归档到 `raw/experiments/YYYY.MM.DD_描述_EXPID/`
5. 日志末尾加「原始材料」段落，引用 raw 路径
6. 检查异常 → 有则追加 `异常记录.md`
7. 追加操作记录到日志索引
8. 告知用户写入位置

模糊信息（温度记不清、样品编号不明）主动询问，不猜测写入。

## 目录结构

```
/vault/
├── raw/experiments/                       ← 原始层（归档）
│   └── YYYY.MM.DD_描述_EXPID/
│       ├── 笔记.md
│       ├── 图片/
│       └── 语音/
│
wiki/实验日志/                              ← 标准层（产出）
├── 实验索引.md
├── 异常记录.md
├── {体系A}/
│   ├── 实验类型1/
│   ├── 实验类型2/
│   └── ...
├── {体系B}/
│   └── ...
└── 公共/
    └── 设备与试剂追踪.md
```

## 实验 ID 规则

```
{体系代码}-{设备代码}-YYMMDD-{序号}
  │        │       │       └─ 当日序号（001 起）
  │        │       └─ 日期
  │        └─ 设备代码（M=马弗炉, T=管式炉, E=电化学, G=手套箱, F=可控气氛炉, B=通用）
  └─ 体系代码（自定义，如 CL / NO / OX / HY 等）
```

## 样品批次 ID 规则

```
{体系代码}-{候选编号}-B{序号}
  │        │         └─ 配盐批次序号
  │        └─ 候选配方编号
  └─ 体系代码
```

同一批样品跨多个实验时 `sample_batch` 保持一致，便于 dataview 追踪。

## 设备代码

| 代码 | 设备 | 场景 |
|------|------|------|
| M | 马弗炉 | 热处理、浸泡腐蚀 |
| T | 管式炉 | 气氛控制、脱水、热稳定性 |
| E | 电化学工作站 | CV/SWV/EIS |
| G | 手套箱 | 配盐、称量、取样 |
| F | 可控气氛炉 | 精密气氛控制 |
| B | 通用 | 干燥、清洗、制样 |

按实际设备扩展。

## Obsidian 集成

本 skill 设计为与 [Obsidian](https://obsidian.md) vault 配合使用。Obsidian 是一个基于本地 Markdown 文件的笔记系统，配合 [Dataview](https://github.com/blacksmithgu/obsidian-dataview) 插件可实现实验数据的动态查询和仪表盘。

**为什么用 Obsidian：**
- 所有日志为纯文本 Markdown，可版本控制、可全文搜索
- YAML frontmatter 结构使 dataview 可自动生成实验列表、异常汇总、设备使用记录
- 本地存储，无云依赖性，数据安全

**安装 skill 后需在 vault 中创建以下文件：**

| 文件 | 模板 | 用途 |
|------|------|------|
| `实验日志/实验索引.md` | `templates/experiment-index.md` | Dataview 查询仪表盘 |
| `实验日志/异常记录.md` | `templates/anomaly-log.md` | 异常记录 |
| `实验日志/公共/设备与试剂追踪.md` | `templates/equipment-tracking.md` | 设备与试剂追踪 |

将模板文件复制到你的 Obsidian vault 对应位置即可使用。

## 参考示例

`references/` 目录包含三个完整的实验日志示例，覆盖常见实验类型：

| 文件 | 实验类型 |
|------|---------|
| `references/example-log.md` | 材料腐蚀浸泡实验 |
| `references/example-electrochemical.md` | 电化学表征（CV 窗口测试） |
| `references/example-thermal-stability.md` | 热稳定性实验 |

每个示例均包含完整的 YAML frontmatter 和 Markdown 正文，可直接作为模板修改使用。

## 飞书 CLI 操作要点

路径 B 使用 `feishu-cli-integration` skill：

- 拉消息：`lark-cli im +chat-messages-list --chat-id oc_*** --page-size 30 --sort asc`
- 下载图片：`lark-cli im +messages-resources-download --message-id *** --file-key *** --type image --output <相对路径>`
- ⚠️ `--output` 只接受相对路径，先 `cd` 到 `raw/experiments/` 归档目录

群 ID 和 bot 权限按 `feishu-cli-integration` skill 的配置获取。

## 自定义指南

- **体系代码**：按你的实验体系自定义（如 CL/NO/OR/PO）
- **实验类型**：在 `wiki/实验日志/{体系}/` 下按需创建子目录
- **YAML 字段**：模板是建议结构，可增删字段
- **设备代码**：按实际实验室设备扩展

## 相关文件

| 文件 | 用途 |
|------|------|
| `references/example-log.md` | 完整实验日志示例 |
| `wiki/实验日志/实验索引.md` | Dataview 仪表盘 |
| `wiki/实验日志/异常记录.md` | 异常记录格式 |
| `wiki/实验日志/公共/设备与试剂追踪.md` | 设备、试剂追踪 |
