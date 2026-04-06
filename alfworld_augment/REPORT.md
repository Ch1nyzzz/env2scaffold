# ALFWorld 环境增强项目进展汇报

## 1. 当前项目框架

目前项目已经形成两层结构：

### 1.1 环境增强 Pipeline

`alfworld_augment/` 负责环境增强的自动化 pipeline，由 `pipeline.py` 串联三个阶段：

1. `Probing Agent`
   - 主动与 ALFWorld 环境交互
   - 收集反馈模式与轨迹数据
   - 产出 `probing/trajectories/` 和 `feedback_catalog.json`

2. `Analysis Agent`
   - 读取 probing 结果和环境源码
   - 生成增强规则
   - 实现运行时 Wrapper `analysis/augmented_env.py`

3. `Verify Agent`
   - 检查增强反馈是否泄露
   - 验证 reward / done / admissible_commands 无回归
   - 输出验证报告

这一层的目标是：把 ALFWorld 的环境反馈增强能力稳定产出为可运行代码。

### 1.2 训练接入层

`AWorld-RL/EnvTuning/` 负责把增强环境接入现有 Agent-RL 训练框架。

当前已经新增了独立的 ALFWorld 训练路径：

- `env_tuning/interaction/alfworld_interaction.py`
- `env_tuning/alfworld_reward.py`
- `env_tuning/config/alfworld_grpo_stage1.yaml`
- `scripts/prepare_alfworld_dataset.py`
- `scripts/run_alfworld_grpo_stage1.sh`

这一层的目标是：把环境增强结果真正接到 GRPO 训练流程中。

## 2. 当前已完成内容

目前已经完成的工作包括：

- 完成 ALFWorld 环境增强 Wrapper
- 完成 probing、analysis、verification 三阶段 pipeline
- 完成增强环境的 clean 验证
- 在增强环境中加入内部 progress 信号：
  - `progress_events`
  - `progress_reward`
  - `progress_score`
  - `progress_milestones`
- 完成 EnvTuning 下的 ALFWorld interaction 接入
- 完成 ALFWorld reward 聚合逻辑
- 完成训练 config、训练脚本和数据准备脚本
- 完成新计算节点部署 checklist
- 完成环境 requirements 文档
- 清理旧 Git 标记，当前仓库已经统一到 monorepo 结构

## 3. 当前产出物

当前可直接使用的核心文件如下：

### 环境增强

- `alfworld_augment/pipeline.py`
- `alfworld_augment/analysis/augmented_env.py`
- `alfworld_augment/verification/verify_runner.py`

### 训练接入

- `AWorld-RL/EnvTuning/env_tuning/interaction/alfworld_interaction.py`
- `AWorld-RL/EnvTuning/env_tuning/alfworld_reward.py`
- `AWorld-RL/EnvTuning/env_tuning/config/alfworld_grpo_stage1.yaml`
- `AWorld-RL/EnvTuning/scripts/prepare_alfworld_dataset.py`
- `AWorld-RL/EnvTuning/scripts/run_alfworld_grpo_stage1.sh`

### 部署文档

- `docs/alfworld_new_node_checklist.md`
- `docs/alfworld_environment_requirements.md`

## 4. 目前状态

目前代码侧准备已经基本完成。

已经具备：

- 可运行的 ALFWorld 环境增强 pipeline
- 可运行的增强环境 Wrapper
- 可供训练使用的 progress reward 信号
- 与 EnvTuning 兼容的训练入口
- 数据准备、配置和启动脚本
- 新节点部署说明

当前阶段不需要重新跑整套环境增强 pipeline，现有增强结果已经可以直接进入训练阶段。

## 5. 下一步

下一步工作是进入训练阶段，在 GPU 节点上完成 ALFWorld GRPO pilot。

计划顺序如下：

1. 在新计算节点部署代码环境和 ALFWorld 运行环境
2. 运行 `prepare_alfworld_dataset.py` 生成训练 parquet 数据
3. 启动 `run_alfworld_grpo_stage1.sh`
4. 完成 baseline 与增强环境的训练对比
5. 观察 success rate、progress return、invalid action 和训练曲线

## 6. 当前汇报结论

当前可以对外汇报为：

> ALFWorld 环境增强 pipeline 已完成，增强环境与 progress reward 已接入 EnvTuning 训练框架，代码和部署文档已准备就绪，下一步是在 GPU 节点上开展 GRPO 训练验证。
