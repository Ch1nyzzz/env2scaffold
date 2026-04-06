# ALFWorld Environment Feedback Augmentation Report

## 1. 问题：为什么需要增强？

ALFWorld 的环境反馈有一个严重缺陷：**所有失败动作都返回同一个字符串 `"Nothing happens."`**。

根源在 TextWorld PDDL 引擎 (`textworld/envs/pddl/pddl.py` ~L162)：

```python
try:
    idx = self.prev_state["_valid_commands"].index(command)
except ValueError:
    self.state.feedback = "Nothing happens."
```

只要命令不在 `_valid_commands` 列表里，无论原因是什么，都返回 `"Nothing happens."`。Agent 完全无法区分以下情况：

| 错误类型 | 示例命令 | 环境反馈（增强前） |
|---------|---------|-----------------|
| 空手放物 | `move plate 1 to shelf 1` | Nothing happens. |
| 从关闭容器取物 | `take apple 1 from fridge 1` | Nothing happens. |
| 物品不在此处 | `take knife 1 from countertop 1` | Nothing happens. |
| 空手加热/冷却/清洗 | `heat apple 1 with microwave 1` | Nothing happens. |
| 重复开/关容器 | `open drawer 1`（已开） | Nothing happens. |
| 手满再拿 | `take book 1 from bed 1`（已拿着另一个） | Nothing happens. |
| 完全无效命令 | `fly to mars` | Nothing happens. |
| 对不可开关容器操作 | `open cabinet 1`（开放式架子） | Nothing happens. |

这意味着 Agent 犯错后没有任何信号来纠正行为，只能盲目重试。

## 2. 探测过程

Probing Agent 系统性地与 ALFWorld 交互，收集了完整的反馈模式：

- **12 个游戏**，覆盖全部 6 种任务类型
- **540 个正确步骤** + **7,957 个错误探测**
- **344 个独特反馈字符串**
- `"Nothing happens."` 出现 7,957 次，对应 **15 种不同原因**

各错误原因分布：

| 原因 | 出现次数 | 占比 |
|------|---------|------|
| 完全无效命令 | 2,160 | 27.1% |
| 空手放物 | 540 | 6.8% |
| 放不存在的物品 | 540 | 6.8% |
| 开不存在的容器 | 540 | 6.8% |
| 取不存在的物品 | 540 | 6.8% |
| 导航到不存在的地点 | 540 | 6.8% |
| 从错误位置取物 | 540 | 6.8% |
| 空手清洗 | 540 | 6.8% |
| 空手加热 | 540 | 6.8% |
| 使用不存在的灯 | 540 | 6.8% |
| 空手冷却 | 539 | 6.8% |
| 开已开容器 | 129 | 1.6% |
| 从关闭容器取物 | 116 | 1.5% |
| 关已关容器 | 116 | 1.5% |
| 未开灯检查 | 37 | 0.5% |

## 3. 增强方案：怎么做的

### 3.1 技术架构

在 gym 层面用一个 **Wrapper 类** (`AugmentedAlfWorldEnv`) 包裹原始环境：

```
Agent <--> AugmentedAlfWorldEnv <--> TextWorld Gym Env <--> PDDL Engine
                |
                ├── 拦截 step() 返回的 obs
                ├── 读取 PDDL facts 获取内部状态
                ├── 匹配增强规则
                └── 替换 obs 文本（不动 reward / done / admissible_commands）
```

关键设计：通过 `request_infos.facts=True` 获取 PDDL 事实（`holds`, `opened`, `openable`, `inreceptacle` 等），构建 `InternalState` 来精确判断失败原因。

### 3.2 状态追踪

Wrapper 在每步 `step()` 前后从 PDDL facts 构建 `InternalState`，追踪：

| 状态维度 | PDDL 事实 | 用途 |
|---------|----------|------|
| 手持物品 | `holds(agent1, X)` | 判断 R01/R06/R07/R08 |
| 容器开关状态 | `opened(X)` + `openable(X)` | 判断 R02/R04/R05 |
| 物品位置 | `inreceptacle(obj, recep)` | 判断 R03 |
| 可操作性 | `openable(X)` | 判断 R04/R05 不可开关容器 |

使用 **pre-step state**（动作执行前的状态）做规则匹配——因为当 `"Nothing happens."` 时，状态没变，pre 和 post 相同；而对成功动作（R10/R11），pre-step 能正确反映动作前的持有/位置状态。

### 3.3 增强规则详情

共 **12 条规则**（11 条原始 + 1 条手动补充），分为三类：

#### 错误诊断规则（替换 "Nothing happens."）

| ID | 触发条件 | 增强反馈 |
|----|---------|---------|
| **R01** | `move X to Y` + 手空 | "You are not holding anything. You need to pick up an object before you can place it somewhere." |
| **R02** | `take X from Y` + Y 是关闭的 | "The {container} is closed. You need to open it first before you can take anything from it." |
| **R03** | `take X from Y` + X 不在 Y 里 | "You cannot find {object} in the {container}. That object is not there. Try looking around other locations." |
| **R04** | `open X` + X 已开 | "The {container} is already open." |
| **R04+** | `open X` + X 不可开关 | "The {container} is not a container that can be opened or closed — its contents are already accessible." |
| **R05** | `close X` + X 已关 | "The {container} is already closed." |
| **R05+** | `close X` + X 不可开关 | 同 R04+ |
| **R06** | `heat/cool/clean X with Y` + 没拿 X | "You are not holding {object}. You need to pick it up before you can {action} it." |
| **R06** | `heat/cool/clean X with Y` + 拿着别的 | "You are not holding {object} (you are holding {held_object}). Pick up {object} first." |
| **R07** | `take X from Y` + 手里已有东西 | "You are already holding {held_object}. You can only carry one object at a time. Put it down first." |
| **R08** | `use lamp` + 手空 | "You are not holding anything. To examine an object under a lamp, you must first pick up the object, then use the lamp." |
| **R09** | 命令不匹配任何已知语法 | "That command is not recognized. Valid actions include: go to, take, move, open, close, examine, use, heat, cool, clean." |

#### 兜底规则

| ID | 触发条件 | 增强反馈 |
|----|---------|---------|
| **Fallback** | `"Nothing happens."` 但无规则匹配 | "Nothing happens. The action could not be performed in the current state. Check your inventory and the state of nearby objects." |

#### 正向引导规则（增强成功反馈）

| ID | 触发条件 | 增强反馈 |
|----|---------|---------|
| **R10** | 拾起了任务目标提到的物品类型 | "{original_obs} [You are now holding the {object_type} needed for this task. Good progress!]" |
| **R11** | 到达空位置 + 手里没有目标物品 | "{original_obs} There is nothing useful here for your current task. Keep exploring." |

### 3.4 规则优先级与互斥

规则按优先级顺序匹配，**第一个匹配的规则生效**，后续不再检查。优先级设计考虑了互斥关系：

```
R01 (空手放物)
 └→ R02 (从关闭容器取物)
      └→ R07 (手满再拿)       ← 必须在 R03 之前：同为 take 命令，但原因不同
           └→ R03 (物品不在此处) ← 排除了 R02（容器关）和 R07（手满）之后
                └→ R04/R05 (重复开关)
                     └→ R06 (空手用电器)
                          └→ R08 (空手用灯)
                               └→ R09 (无效命令) ← 最低优先级，兜底语法错误
```

## 4. 不做什么（设计约束）

为确保增强不影响任务难度的公平性，严格遵守以下约束：

| 约束 | 说明 |
|------|------|
| **不修改 reward** | `score` 原样透传 |
| **不修改 done** | 终止信号原样透传 |
| **不修改 admissible_commands** | 合法动作列表原样透传 |
| **不泄露解法** | 不说"去 countertop 2 拿 plate"，只说"你手空了，先拿东西" |
| **不过度约束探索** | 提示方向但不限定唯一路径 |
| **反馈信息来源只用 agent 自己的命令** | R02 说"The cabinet 3 is closed"——cabinet 3 来自 agent 自己输入的命令，不是从内部状态泄露的 |

## 5. 验证结果

### 泄露检查：11/11 通过

所有规则均无解法泄露。边界case分析：
- R03 提到 `{object}` 和 `{container}`：来自 agent 自己的命令，不是内部状态
- R06 提到 `{held_object}`：agent 自己之前 pick up 的，是状态提醒不是泄露
- R10 确认"needed for this task"：物品类型已出现在公开的任务描述中

### 无回归测试：5/5 通过

在 5 个完整 episode 上对比 original vs augmented：
- reward 完全一致
- done 信号完全一致
- admissible_commands 完全一致
- Wrapper 对核心游戏逻辑完全透明

### 错误恢复测试：9/9 有效

每条错误规则的增强反馈都指向 `admissible_commands` 中存在的有效恢复动作。

## 6. 使用方式

```python
import textworld, textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from analysis.augmented_env import AugmentedAlfWorldEnv

# 创建 base env（注意 facts=True 是必须的）
request_infos = textworld.EnvInfos(
    won=True, admissible_commands=True, facts=True, extras=["gamefile"]
)
env_id = textworld.gym.register_games(
    game_files, request_infos,
    batch_size=1, asynchronous=False,
    max_episode_steps=50,
    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]
)
base_env = textworld.gym.make(env_id)

# 包裹增强环境
env = AugmentedAlfWorldEnv(base_env, verbose=False)

# 正常使用
obs, infos = env.reset()
obs, score, done, infos = env.step("go to cabinet 1")

# 查看增强日志
for entry in env.augmentation_log:
    print(f"Step {entry['episode_step']}: [{entry['rule_applied']}] {entry['augmented_obs']}")
```

## 7. 文件清单

```
alfworld_augment/
├── pipeline.py                              # 主 orchestrator
├── prompts/
│   ├── probing_agent_prompt.md              # Probing Agent 指令
│   ├── analysis_agent_prompt.md             # Analysis Agent 指令
│   └── verify_agent_prompt.md               # Verify Agent 指令
├── probing/
│   ├── probe_runner.py                      # 探测脚本 (846行)
│   ├── trajectories/                        # 12 个轨迹 JSON
│   └── feedback_catalog.json                # 反馈模式目录
├── analysis/
│   ├── source_analysis.md                   # TextWorld 源码分析
│   ├── augmentation_plan.json               # 增强规则定义
│   ├── augmented_env.py                     # 增强环境 Wrapper (558行)
│   └── smoke_test_result.json               # 冒烟测试结果 (14/14 通过)
└── verification/
    ├── verify_runner.py                     # 验证脚本
    ├── verify_results.json                  # 验证数据
    └── verify_report.md                     # 验证报告
```

## 8. 目前进展

在完成 ALFWorld 环境反馈增强 MVP 后，当前代码已经进一步接入训练框架，整体进展如下：

- 已完成环境增强 Wrapper，运行时文件为 `analysis/augmented_env.py`
- 已补充内部 progress 信号，包括：
  - `progress_events`
  - `progress_reward`
  - `progress_score`
  - `progress_milestones`
- 已完成 clean 行为验证，确认 progress 信号保留在 `infos` 中，不暴露给模型 observation
- 已在 `AWorld-RL/EnvTuning` 中新增 ALFWorld 训练接入，包括：
  - `env_tuning/interaction/alfworld_interaction.py`
  - `env_tuning/alfworld_reward.py`
  - `env_tuning/config/alfworld_grpo_stage1.yaml`
  - `scripts/prepare_alfworld_dataset.py`
  - `scripts/run_alfworld_grpo_stage1.sh`
- 已完成新计算节点部署文档：
  - `docs/alfworld_new_node_checklist.md`
  - `docs/alfworld_environment_requirements.md`
- 已清理旧的 Git 子模块标记，当前仓库为统一 monorepo 结构

目前代码侧已经具备：

- ALFWorld 环境增强
- 细粒度 progress reward
- EnvTuning 训练接入
- 数据准备脚本
- 训练配置和启动脚本

## 9. 下一步

下一步工作是进入训练阶段，在 GPU 节点上完成 ALFWorld GRPO pilot。

计划顺序如下：

1. 按 `docs/alfworld_new_node_checklist.md` 部署新计算节点环境
2. 运行 `python scripts/prepare_alfworld_dataset.py` 生成 parquet 数据
3. 使用 `bash scripts/run_alfworld_grpo_stage1.sh` 启动小规模训练
4. 对比 baseline 环境和增强环境的训练表现
5. 观察 success rate、progress return、invalid action 和训练曲线

当前汇报可以概括为：

> ALFWorld 环境增强与 progress reward 的工程接入已经完成，训练准备工作已基本就绪，下一步是在 GPU 节点上开展 GRPO 训练验证。
