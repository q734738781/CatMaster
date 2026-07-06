# experiment-log

标准化实验日志记录工具。接收图片、语音、文字等原始材料，自动生成带 YAML frontmatter 的结构化日志。

## 这是什么

不是"帮我记实验"——它是一套标准化的日志管线：接收原始材料 → 提取结构化信息 → 生成唯一 ID → 写出标准格式日志 → 归档原始文件。

适用于任何需要规范实验记录的研究领域。

## 安装

```bash
git clone https://github.com/Jiahao8595/research-pipeline.git
cp -r research-pipeline/experiment-log ~/.hermes/skills/
```

安装后 `/reload-skills`。

**前置依赖：**

```bash
hermes skills install feishu-cli-integration   # 飞书消息接收（路径 B）
hermes skills install obsidian                 # vault 文件管理
```

## 使用方式

**路径 A — CLI 直接提交：**
```
"记录一个实验：今天在 500°C 做了 316L 的氯盐腐蚀，300h，Ar 气氛，失重 0.0032g"
```
附上图片，agent 会 vision_analyze 提取信息并生成标准日志。

**路径 B — 飞书提交：**
把实验照片和语音发到配置好的飞书群，agent 自动扫描并处理。

## 文件结构

```
experiment-log/
├── SKILL.md                    ← 技能入口
├── README.md                   ← 本文件
└── references/
    └── example-log.md          ← 完整日志示例
```

## 核心设计

- **统一 ID 体系** — 体系代码 + 设备代码 + 日期 + 序号，跨实验可追踪
- **样品批次追踪** — 同一批样品 ID 一致，dataview 可查
- **YAML frontmatter** — 结构化数据 + 自由文本，既人可读又机器可查
- **飞书 CLI 集成** — 手机拍照发群 → 自动入 vault
- **异常追踪** — 自动检测异常并追加到异常记录

## 自定义

设备代码、体系代码、实验类型目录全部可按你的实验室配置。详见 SKILL.md 中的「自定义指南」。

## 作者

十五 (JL Lab)
