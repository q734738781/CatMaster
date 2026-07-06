# `_shared/` - nature-* 技能的公共内容

这个目录**不是一个独立技能**。它没有 `SKILL.md`，也不会被插件加载器注册。它的作用是存放多个 `nature-*` 技能共同依赖的参考内容，避免在不同技能目录中重复维护同一套材料。

同级技能会通过 `manifest.yaml` 中的相对路径引用这里的文件，例如：

```yaml
always_load:
  - ../_shared/core/reader-workflow.md
```

## 当前内容

| 文件 | 使用方 |
|---|---|
| `core/reader-workflow.md` | `nature-polishing`, `nature-writing` |
| `core/paper-type-taxonomy.md` | `nature-polishing`, `nature-writing` |
| `core/ethics.md` | `nature-polishing`, `nature-writing` |
| `core/terminology-ledger.md` | `nature-polishing`, `nature-writing` |
| `journal-formats/nat-comms.md` | `nature-polishing`, `nature-writing` |

## 什么时候把文件放到这里

只有当**两个或更多技能**需要复用同一份内容时，才把文件放入 `_shared/`。如果内容只服务于一个技能，应保留在该技能自己的 `static/` 或 `references/` 目录中。

## 什么时候保持技能内局部内容

共享层只放**定义和参考材料**，例如论文类型分类、读者工作流、伦理规则或术语表。具体技能如何诊断、起草、修改或输出结果，仍应保留在各自的 `static/fragments/` 中。多个技能可以复用同一套论文类型分类，但在其上执行不同的任务逻辑。
