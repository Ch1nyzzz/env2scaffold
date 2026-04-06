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

## 4. 增强方式概述

### 4.1 `Nothing happens.` 的增强方式

ALFWorld 原始环境中，大量失败动作都会统一返回 `Nothing happens.`。  
当前做法是在 `analysis/augmented_env.py` 中包一层 Wrapper，在每次 `step()` 后：

- 读取环境返回的 `facts` 和 `admissible_commands`
- 构造内部状态，例如：
  - 当前是否持有物品
  - 容器是否打开
  - 物品当前位于哪个 receptacle
- 根据动作类型和内部状态匹配规则
- 将模糊失败反馈替换为更具体、可操作的反馈

例如会区分出：

- 手里没有东西却执行 `move`
- 容器没开却执行 `take`
- 已经拿着别的物体又继续 `take`
- 命令格式本身无效

这样环境不再只返回统一报错，而是返回带原因的反馈文本。

简化示例如下：

- 原始反馈：
  - `move plate 1 to shelf 1`
  - `Nothing happens.`
- 增强反馈：
  - `You are not holding anything. You need to pick up an object before you can place it somewhere.`

- 原始反馈：
  - `take apple 1 from fridge 1`
  - `Nothing happens.`
- 增强反馈：
  - `The fridge 1 is closed. You need to open it first before you can take anything from it.`

- 原始反馈：
  - `take book 1 from bed 1`（手里已经拿着别的物品）
  - `Nothing happens.`
- 增强反馈：
  - `You are already holding vase 1. You can only carry one object at a time. Put it down first.`

### 4.2 Progress Reward 的构造方式

当前 progress reward 也是在 `analysis/augmented_env.py` 中构造，并只通过 `infos` 返回给训练框架。

构造方式是：

- 先从任务描述中解析任务目标
  - 目标物体类型
  - 目标位置类型
  - 任务类型
- 再结合当前 observation、内部状态和历史访问记录
- 识别 milestone 事件

目前使用的 milestone 包括：

- `found_target_object`
- `found_target_destination`
- `holding_target_object`
- `placed_target_object`
- `completed_light_inspection`
- `task_completed`

同时也对明显空转行为给轻微负反馈，例如：

- `revisited_empty_location`

每步会输出：

- `progress_events`
- `progress_reward`
- `progress_score`
- `progress_milestones`

这些字段不会暴露给模型，只供 trainer 和 evaluator 使用。

简化示例如下：

- 当 agent 首次走到目标物附近并看到目标物时：
  - `progress_events = ["found_target_object"]`
  - `progress_reward = +0.5`

- 当 agent 成功拿起目标物时：
  - `progress_events = ["holding_target_object"]`
  - `progress_reward = +1.0`

- 当 agent 把目标物成功放到目标位置时：
  - `progress_events = ["placed_target_object"]`
  - `progress_reward = +2.0`

- 当 agent 反复回到没有任务相关物体的空位置时：
  - `progress_events = ["revisited_empty_location"]`
  - `progress_reward = -0.05`

## 5. 目前状态

目前代码侧准备已经基本完成。

已经具备：

- 可运行的 ALFWorld 环境增强 pipeline
- 可运行的增强环境 Wrapper
- 可供训练使用的 progress reward 信号
- 与 EnvTuning 兼容的训练入口
- 数据准备、配置和启动脚本
- 新节点部署说明

当前阶段不需要重新跑整套环境增强 pipeline，现有增强结果已经可以直接进入训练阶段。

## 6. 下一步

下一步工作是进入训练阶段，在 GPU 节点上完成 ALFWorld GRPO pilot。

计划顺序如下：

1. 在新计算节点部署代码环境和 ALFWorld 运行环境
2. 运行 `prepare_alfworld_dataset.py` 生成训练 parquet 数据
3. 启动 `run_alfworld_grpo_stage1.sh`
4. 完成 baseline 与增强环境的训练对比
5. 观察 success rate、progress return、invalid action 和训练曲线

## 7. 当前汇报结论

当前可以对外汇报为：

> ALFWorld 环境增强 pipeline 已完成，增强环境与 progress reward 已接入 EnvTuning 训练框架，代码和部署文档已准备就绪，下一步是在 GPU 节点上开展 GRPO 训练验证。
