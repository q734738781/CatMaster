# CatMaster LLM Agent 系统架构

## 系统概览

```
┌─────────────────┐
│   用户请求      │ 
│  (自然语言)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  规划 Agent     │◀─────│   ChatGPT    │
│ (PlanningAgent) │      │  (GPT-4)     │
└────────┬────────┘      └──────────────┘
         │
         │ 计算计划
         ▼
┌─────────────────┐      ┌──────────────┐
│  编排 Agent     │◀─────│  LangGraph   │
│(OrchestratorAgent)│     │  (工作流)    │
└────────┬────────┘      └──────────────┘
         │
         │ 任务分发
         ▼
┌─────────────────┐
│  执行 Agent     │
│(ExecutionAgent) │
└────────┬────────┘
         │
    ┌────┴────┬─────────┐
    ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐
│ Local  ││  GPU   ││  CPU   │
│ 文件准备││ Worker ││ Worker │
└────────┘└────────┘└────────┘
    │         │         │
    │    ┌────┴────┐    │
    │    │  MACE   │    │
    │    │ 优化    │    │
    │    └─────────┘    │
    │                   │
    │              ┌────┴────┐
    │              │  VASP   │
    │              │  计算   │
    │              └─────────┘
    │                   │
    └────────┬──────────┘
             │
             ▼
    ┌─────────────────┐
    │   总结 Agent    │
    │ (SummaryAgent)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   计算报告      │
    │  - 能量        │
    │  - 结构        │  
    │  - 分析        │
    └─────────────────┘
```

## 数据流

### 1. 用户输入阶段
```
用户: "请计算O2分子的优化结构和精确能量"
  ↓
规划Agent: 解析意图，生成计算计划
```

### 2. 计算计划
```json
{
  "objective": "计算O2分子的优化结构和精确能量",
  "steps": [
    {
      "name": "prepare_structure",
      "method": "create_molecule",
      "device": "local"
    },
    {
      "name": "mace_preopt", 
      "method": "mace_relax",
      "device": "gpu-worker"
    },
    {
      "name": "vasp_prepare",
      "method": "vasp_prepare", 
      "device": "local"
    },
    {
      "name": "vasp_calculate",
      "method": "vasp_execute",
      "device": "cpu-worker"
    }
  ]
}
```

### 3. 任务执行流程

#### Local执行
- 创建O2分子初始结构 (POSCAR)
- 准备VASP输入文件

#### GPU Worker执行  
- SFTP上传结构文件
- 运行MACE力场优化
- 下载优化后的结构

#### CPU Worker执行
- SFTP上传VASP输入
- 提交SLURM作业
- 运行VASP计算
- 下载结果文件

### 4. 结果汇总
```
总结Agent收集所有结果:
- MACE能量: -X.XX eV
- VASP能量: -Y.YY eV  
- O-O键长: 1.21 Å
- 计算时间: XX分钟
```

## 关键组件

### LLM执行适配器 (llm_adapter.py)
封装jobflow-remote执行流程:
- 文件上传 (SFTP)
- 作业提交
- 状态监控
- 结果下载

### Agent间通信
使用LangGraph的StateGraph管理:
- 共享状态 (WorkflowState)
- 条件转移
- 错误处理
- 重试机制

### 错误处理策略
1. 任务级重试 (最多3次)
2. 失败后继续执行其他任务
3. 部分结果也生成报告
4. 详细错误日志

## 扩展性设计

### 添加新的计算方法
1. 在`catmaster/tools/`实现工具函数
2. 在`ExecutionAgent`添加调用
3. 更新`PlanningAgent`提示词

### 支持新的分子体系
1. 扩展`create_molecule`方法
2. 调整计算参数
3. 更新结果分析逻辑

### 集成新的Worker
1. 配置jobflow-remote
2. 实现对应的执行逻辑
3. 更新资源分配策略
