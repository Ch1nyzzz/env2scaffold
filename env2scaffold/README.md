# Benchmark2Scaffold

**给定任意 Agent Benchmark，自动生成环境反馈增强方案。**

基于论文 [*"Don't Just Fine-tune the Agent, Tune the Environment"*](https://arxiv.org/abs/2510.10197) 的思路，我们构建了一套全自动 pipeline，通过 Claude Code 驱动三个 sub-agent 协作完成环境反馈增强。当前 MVP 已在 ALFWorld 上验证通过。

---

## 目录

- [1. 动机与核心思路](#1-动机与核心思路)
- [2. Pipeline 架构](#2-pipeline-架构)
- [3. ALFWorld MVP：完整案例](#3-alfworld-mvp完整案例)
  - [3.1 问题分析](#31-问题分析)
  - [3.2 Phase 1: Probing Agent](#32-phase-1-probing-agent)
  - [3.3 Phase 2: Analysis Agent](#33-phase-2-analysis-agent)
  - [3.4 Phase 3: Verify Agent](#34-phase-3-verify-agent)
- [4. 增强规则详解](#4-增强规则详解)
- [5. 验证结果](#5-验证结果)
- [6. 使用方式](#6-使用方式)
- [7. 项目结构](#7-项目结构)

---

## 1. 动机与核心思路

### 问题

当前 Agent Benchmark（ALFWorld、ScienceWorld、WebShop、TAU-bench 等）的环境反馈普遍存在一个问题：**错误反馈信息量不足**。Agent 犯错后得到的反馈往往过于模糊，无法帮助 Agent 理解错误原因和纠正方向。

以 ALFWorld 为例，所有失败动作——无论原因是什么——都返回同一个字符串：

```
> put plate 1 in/on shelf 1    (空手)        → "Nothing happens."
> take knife from fridge 1      (fridge 关着)  → "Nothing happens."
> fly to mars                   (无效命令)     → "Nothing happens."
```

这 15 种完全不同的错误原因，Agent 看到的反馈完全一样。

### 论文方法的局限

论文 (EnvAug) 提出了修改环境反馈来增强 Agent 训练效果，但其方法是**半自动**的：
- 需要先用一个 "bad model" 与环境交互来收集失败轨迹
- 需要人工为每个领域提供 3-5 个 few-shot example
- 仅在 BFCL 工具调用场景上验证

### 我们的方法

核心洞察：**我们不需要 bad model**。环境反馈是环境的属性，不是 Agent 的属性。我们可以让自己的 sub-agent **主动系统性地探测**环境——故意做对、做错——从而直接收集所有反馈模式。

这比论文方法更好：
- **全自动**：不需要人工 few-shot example
- **覆盖率更高**：系统性探测 vs 依赖 bad model 随机碰到的错误
- **通用性更强**：pipeline 可以应用于任意 Gym 兼容 benchmark

---

## 2. Pipeline 架构

整个 pipeline 由一个 Python orchestrator 驱动三个 Claude Code sub-agent 顺序执行：

```
                     ┌────────────────────────┐
                     │  pipeline.py           │
                     │  (Python Orchestrator) │
                     └──────┬──┬──┬──────────┘
                            │  │  │
            ┌───────────────┘  │  └───────────────┐
            ▼                  ▼                   ▼
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ Probing Agent  │ │ Analysis Agent │ │ Verify Agent   │
   │                │ │                │ │                │
   │ 主动探测环境   │ │ 读源码+轨迹   │ │ 检查泄露       │
   │ 收集反馈模式   │→│ 设计增强方案   │→│ 验证有效性     │
   │                │ │ 实现 Wrapper   │ │ 无回归测试     │
   │                │ │ Smoke Test     │ │                │
   └────────────────┘ └────────────────┘ └────────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
   trajectories/       augmented_env.py      verify_report.md
   feedback_catalog    augmentation_plan
```

### Orchestrator (`pipeline.py`)

Orchestrator 的职责：
1. **按顺序启动** sub-agent：Probing → Analysis → Verify
2. **Phase gating**：每个 agent 结束后，检查预期输出文件是否存在且非空
3. **日志记录**：每个 agent 的 stdout 实时写入 `{agent_name}.log`

每个 sub-agent 通过 Claude Code CLI 的 headless 模式启动：

```python
cmd = [
    "claude",
    "-p",                            # print 模式（非交互）
    "--dangerously-skip-permissions", # 自动批准所有工具调用
    "--model", "sonnet",             # 使用 Sonnet 节省成本
    "--system-prompt-file", prompt,  # agent 的完整指令
    "Execute the task...",           # 触发执行
]
```

### 数据流

Agent 之间通过文件系统传递数据：

```
Probing Agent 写入:
  probing/trajectories/*.json      ← 每个游戏的完整探测轨迹
  probing/feedback_catalog.json    ← 汇总的反馈模式目录
                    │
                    ▼
Analysis Agent 读取上述文件 + ALFWorld 源码，写入:
  augmentation/source_analysis.md      ← 源码分析报告
  augmentation/augmentation_plan.json  ← 增强规则定义
  augmentation/augmented_env.py        ← 环境增强 Wrapper
  augmentation/smoke_test_result.json  ← 冒烟测试结果
                    │
                    ▼
Verify Agent 读取所有上述文件，写入:
  verification/verify_report.md    ← 完整验证报告
```

---

## 3. ALFWorld MVP：完整案例

### 3.1 问题分析

ALFWorld 是一个基于 TextWorld 的文本交互式家庭环境，包含 6 种任务类型：

| 任务类型 | 目标 |
|---------|------|
| pick_and_place_simple | 找到物品 → 拿起 → 放到目标位置 |
| look_at_obj_in_light | 找到物品 → 找到灯 → 开灯 → 检查物品 |
| pick_clean_then_place | 找到物品 → 到水槽清洗 → 放到目标位置 |
| pick_heat_then_place | 找到物品 → 用微波炉加热 → 放到目标位置 |
| pick_cool_then_place | 找到物品 → 用冰箱冷却 → 放到目标位置 |
| pick_two_obj_and_place | 找到两个同类物品 → 放到目标位置 |

**反馈生成机制**位于 TextWorld PDDL 引擎的最底层 (`textworld/envs/pddl/pddl.py`)：

```python
def step(self, command):
    try:
        idx = self.prev_state["_valid_commands"].index(command)
        # 执行动作...
    except ValueError:
        # 命令不在合法动作列表里 → 统一返回
        self.state.feedback = "Nothing happens."
```

所有失败动作——无论原因——都经过这同一个 `except ValueError` 分支，返回同一个字符串。

### 3.2 Phase 1: Probing Agent

**目标**：系统性地与环境交互，收集完整的反馈模式。

**做了什么**：

Probing Agent 编写并执行了一个 846 行的探测脚本 (`probe_runner.py`)。对每个游戏实例，脚本执行两种操作：

**A. 正确路径探索**：沿 `admissible_commands`（环境提供的合法命令列表）执行正确动作，记录每步的 observation、score、done 状态。

**B. 系统性错误探测**：在每个正确步骤前，额外执行一系列故意错误的动作。错误动作按 6 个类别设计：

| 类别 | 探测内容 | 示例 |
|------|---------|------|
| 库存错误 | 空手放物、手满再拿 | `put plate 1 in/on shelf 1`（空手） |
| 容器状态错误 | 从关闭容器取物、重复开关 | `take apple from fridge 1`（fridge 关着） |
| 位置错误 | 取不存在的物品、去不存在的地点 | `go to nonexistent 99` |
| 电器操作错误 | 空手加热/冷却/清洗 | `heat apple with microwave 1`（空手） |
| 灯光/检查错误 | 未开灯检查、使用不存在的灯 | `use desklamp 99` |
| 完全无效命令 | 语法不合法的命令 | `fly to mars` |

**产出**：

- **12 个轨迹文件**（每种任务类型 2 个），每个约 600-700KB
- **7,957 个错误探测**，覆盖 15 种不同的错误原因
- `feedback_catalog.json`：汇总了所有 344 个独特反馈字符串及其触发条件

**关键发现**：所有 7,957 个错误探测返回的反馈都是同一个字符串 `"Nothing happens."`，对应 15 种不同的错误原因：

| 错误原因 | 出现次数 |
|---------|---------|
| 完全无效命令 (`fly to mars`) | 2,160 |
| 空手放物 | 540 |
| 放不存在的物品 | 540 |
| 开不存在的容器 | 540 |
| 取不存在的物品 | 540 |
| 去不存在的地点 | 540 |
| 从错误位置取物 | 540 |
| 空手清洗/加热/冷却 | 各 ~540 |
| 开已开容器 | 129 |
| 关已关容器 / 从关闭容器取物 | 各 116 |
| 未开灯检查 | 37 |

### 3.3 Phase 2: Analysis Agent

**目标**：分析源码 + 探测数据 → 设计增强方案 → 实现 Wrapper → 冒烟测试。

**做了什么**：

#### Step 1: 源码分析

Analysis Agent 深入阅读了 TextWorld 和 ALFWorld 的源码，产出了 `source_analysis.md`。关键发现：

1. **`"Nothing happens."` 的生成位置**：TextWorld PDDL 引擎最底层，`pddl.py` 的 `step()` 方法中，当命令不在 `_valid_commands` 列表里时硬编码返回。

2. **环境内部可用但未暴露给 Agent 的信息**：通过设置 `request_infos.facts=True`，可以获取 PDDL 事实，包括：
   - `holds(agent1, X)` — Agent 手里拿着什么
   - `opened(X)` — 哪些容器是打开的
   - `openable(X)` — 哪些容器可以开关
   - `inreceptacle(obj, recep)` — 物品在哪个容器里

3. **这些信息足以精确判断每种失败原因**。

#### Step 2: 增强方案设计

基于源码分析和探测数据的交叉分析，设计了 11 条增强规则（后手动补充 1 条，共 12 条）。每条规则定义了：

- **触发条件**：observation 内容 + 命令模式 + 内部状态条件
- **检测方法**：如何从 PDDL facts 判断这个条件成立
- **增强反馈**：替换原始 `"Nothing happens."` 的文本
- **泄露风险评估**：这条反馈是否可能泄露解法

#### Step 3: Wrapper 实现

实现了 `AugmentedAlfWorldEnv` 类（558 行），核心设计：

**状态追踪**：Wrapper 在每步 `step()` 前后从 PDDL facts 构建 `InternalState` 对象：

```python
class InternalState:
    held_object: str          # 手持物品（从 holds fact 获取）
    open_containers: set      # 已打开的容器（从 opened facts 获取）
    openable_containers: set  # 可开关的容器（从 openable facts 获取）
    object_in_receptacle: dict  # 物品→容器映射（从 inreceptacle facts 获取）
```

**规则匹配**：当 observation 为 `"Nothing happens."` 时，按优先级顺序检查每条规则。使用 **pre-step state**（动作执行前的状态）做匹配——因为失败时状态不变，而成功规则（R10/R11）需要知道动作前的状态。

**透明代理**：Wrapper 仅修改 observation 文本，**不修改** reward、done、admissible_commands 中的任何一个。

#### Step 4: 冒烟测试

编写并运行了专项测试脚本，**14/14 全部通过**：

| 测试 | 触发命令 | 预期规则 | 结果 |
|------|---------|---------|------|
| 空手放物 | `move book 1 to sidetable 1` | R01 | "You are not holding anything..." |
| 从关闭容器取物 | `take cellphone 1 from drawer 1` | R02 | "The drawer 1 is closed..." |
| 物品不在此处 | `take pillow 1 from sidetable 1` | R03 | "You cannot find pillow 1..." |
| 开已开容器 | `open drawer 1`（已开） | R04 | "The drawer 1 is already open." |
| 关已关容器 | `close drawer 1`（已关） | R05 | "The drawer 1 is already closed." |
| 空手加热 | `heat apple 1 with microwave 1` | R06 | "You are not holding anything..." |
| 空手冷却 | `cool apple 1 with fridge 1` | R06 | "You are not holding anything..." |
| 空手清洗 | `clean butterknife 1 with sinkbasin 1` | R06 | "You are not holding anything..." |
| 手满再拿 | `take book 2 from bed 1`（已拿 book 1） | R07 | "You are already holding book 1..." |
| 空手用灯 | `use desklamp 1` | R08 | "You are not holding anything..." |
| 无效命令 | `fly to mars` | R09 | "That command is not recognized..." |
| 拿到目标物品 | `take apple 1 from diningtable 2` | R10 | "...holding the apple needed for this task..." |
| 到达空位置 | `go to sidetable 1` | R11 | "...nothing useful here...Keep exploring." |
| 不变性检查 | `go to bed 1`（正常动作） | 无 | 正常 obs，score=0，done=false |

### 3.4 Phase 3: Verify Agent

**目标**：独立验证增强方案的安全性和有效性。

**做了什么**：

#### 泄露检查：11/11 通过

逐条审查每条规则的增强反馈，确认无解法泄露。对边界 case 的分析：

- **R02** 说 "The drawer 1 is closed"：`drawer 1` 来自 Agent 自己输入的命令，不是内部状态泄露
- **R06** 说 "You are not holding apple 1"：`apple 1` 来自 Agent 自己输入的命令
- **R07** 说 "You are already holding book 1"：Agent 自己执行了 pick up 动作，这是状态提醒不是泄露
- **R10** 确认 "needed for this task"：物品类型已出现在公开可见的任务描述字符串中

#### 无回归测试：5/5 通过

在 5 个完整 episode 上逐步对比 original vs augmented 环境：
- **reward 完全一致**
- **done 信号完全一致**
- **admissible_commands 完全一致**
- Wrapper 对核心游戏逻辑完全透明

#### 错误恢复测试：9/9 有效

对每条错误规则，验证增强反馈指向的恢复方向是否在 `admissible_commands` 中存在。例如：
- R02 说 "open it first" → 确认 `open drawer 1` 在 admissible_commands 中
- R01 说 "pick up an object" → 确认 `take X from Y` 在 admissible_commands 中

---

## 4. 增强规则详解

### 错误诊断规则（替换 "Nothing happens."）

| ID | 场景 | 检测方式 | 增强反馈 |
|----|------|---------|---------|
| **R01** | 空手放物 | `move X to Y` + 无 `holds` fact | "You are not holding anything. You need to pick up an object before you can place it somewhere." |
| **R02** | 从关闭容器取物 | `take X from Y` + Y 有 `openable` 无 `opened` | "The {container} is closed. You need to open it first before you can take anything from it." |
| **R03** | 物品不在此处 | `take X from Y` + X 的 `inreceptacle` 指向别处 | "You cannot find {object} in the {container}. That object is not there. Try looking around other locations." |
| **R04** | 开已开容器 | `open X` + X 有 `opened` fact | "The {container} is already open." |
| **R04+** | 开不可开容器 | `open X` + X 无 `openable` fact | "The {container} is not a container that can be opened or closed — its contents are already accessible." |
| **R05** | 关已关容器 | `close X` + X 有 `openable` 无 `opened` | "The {container} is already closed." |
| **R05+** | 关不可关容器 | `close X` + X 无 `openable` fact | 同 R04+ |
| **R06** | 空手用电器 | `heat/cool/clean X with Y` + 无 `holds` | "You are not holding anything. You need to pick up {object} before you can {action} it." |
| **R06** | 拿错物品用电器 | 同上 + `holds` 不匹配 | "You are not holding {object} (you are holding {held_object}). Pick up {object} first." |
| **R07** | 手满再拿 | `take X from Y` + 有 `holds` fact + Y 未关 | "You are already holding {held_object}. You can only carry one object at a time. Put it down first." |
| **R08** | 空手用灯 | `use X` + 无 `holds` | "You are not holding anything. To examine an object under a lamp, you must first pick up the object, then use the lamp." |
| **R09** | 无效命令 | 命令不匹配任何已知语法模式 | "That command is not recognized. Valid actions include: go to, take, move, open, close, examine, use, heat, cool, clean." |
| **Fallback** | 未匹配任何规则 | `"Nothing happens."` 但所有规则都不匹配 | "Nothing happens. The action could not be performed in the current state. Check your inventory and the state of nearby objects." |

### 正向引导规则（增强成功反馈）

| ID | 场景 | 检测方式 | 增强反馈 |
|----|------|---------|---------|
| **R10** | 拿到任务目标物品 | obs 匹配 `"You pick up the X"` + X 类型出现在任务描述中 | "{original_obs} [You are now holding the {object_type} needed for this task. Good progress!]" |
| **R11** | 到达空位置 | obs 包含 `"you see nothing"` + 以 `"you arrive"` 开头 + 手空 | "{original_obs} There is nothing useful here for your current task. Keep exploring." |

### 规则优先级

规则按优先级顺序匹配，第一个命中的生效。优先级考虑了互斥关系：

```
R01 (空手放物)
 → R02 (从关闭容器取物)
    → R07 (手满再拿)       ← 必须在 R03 之前
       → R03 (物品不在此处) ← 排除 R02 和 R07 之后才检查
          → R04/R05 (重复开关 + 不可开关)
             → R06 (空手用电器)
                → R08 (空手用灯)
                   → R09 (无效命令) ← 最低优先级兜底
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **不修改 reward** | score 原样透传，不影响训练信号 |
| **不修改 done** | 终止信号原样透传 |
| **不修改 admissible_commands** | 合法动作列表原样透传 |
| **不泄露解法** | 不说 "去 countertop 2 拿 plate"，只说 "你手空了，先拿东西" |
| **不过度约束探索** | 提示方向但不限定唯一路径 |
| **反馈中引用的对象名仅来自 Agent 自己的命令** | R02 说 "The cabinet 3 is closed"——cabinet 3 来自 Agent 输入的命令 |

---

## 5. 验证结果

| 检查项 | 结果 |
|--------|------|
| 泄露检查（11 条规则） | 11/11 通过 |
| 冒烟测试（14 项） | 14/14 通过 |
| 无回归测试（5 个 episode） | reward/done/admissible_commands 完全一致 |
| 错误恢复测试（9 条规则） | 9/9 恢复提示有效 |

---

## 6. 使用方式

### 运行完整 Pipeline

```bash
cd env2scaffold
python pipeline.py                      # 全量运行三个 agent
python pipeline.py --agent probing      # 只运行 Probing Agent
python pipeline.py --agent analysis     # 只运行 Analysis Agent（需要 probing 产出）
python pipeline.py --agent verify       # 只运行 Verify Agent（需要 analysis 产出）
python pipeline.py --resume analysis    # 从 Analysis Agent 开始
```

### 直接使用增强环境

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
    batch_size=1, asynchronous=False, max_episode_steps=50,
    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]
)
base_env = textworld.gym.make(env_id)

# 包裹增强环境
env = AugmentedAlfWorldEnv(base_env, verbose=False)

# 正常使用（API 完全兼容）
obs, infos = env.reset()
obs, score, done, infos = env.step("go to cabinet 1")

# 查看增强日志
for entry in env.augmentation_log:
    print(f"Step {entry['episode_step']}: [{entry['rule_applied']}]")
    print(f"  Original: {entry['original_obs']}")
    print(f"  Augmented: {entry['augmented_obs']}")
```

### 增强前后对比示例

```
任务：clean some ladle and put it in countertop.

Step 1: go to cabinet 1
  → "You arrive at cabinet 1. On the cabinet 1, you see a bowl 1."  (不变)

Step 2: put plate 1 in/on shelf 1  (空手放物)
  增强前: "Nothing happens."
  增强后: "You are not holding anything. You need to pick up an object
           before you can place it somewhere."

Step 3: fly to mars  (无效命令)
  增强前: "Nothing happens."
  增强后: "That command is not recognized. Valid actions include: go to,
           take, move, open, close, examine, use, heat, cool, clean."

Step 4: clean plate 1 with sinkbasin 1  (拿着 bowl 1 想清洗 plate 1)
  增强前: "Nothing happens."
  增强后: "You are not holding plate 1 (you are holding bowl 1).
           Pick up plate 1 first."
```

---

## 7. 项目结构

```
env2scaffold/
├── README.md                                   # 本文档
├── REPORT.md                                   # 技术报告（中文）
├── pipeline.py                                 # Pipeline orchestrator
│
├── prompts/                                    # Sub-agent 指令
│   ├── probing_agent_prompt.md                 # Probing Agent 的完整任务描述
│   ├── analysis_agent_prompt.md                # Analysis Agent 的完整任务描述
│   └── verify_agent_prompt.md                  # Verify Agent 的完整任务描述
│
├── probing/                                    # Phase 1 产出
│   ├── probe_runner.py                         # 探测脚本（846 行）
│   ├── trajectories/                           # 12 个轨迹 JSON（每种任务 2 个）
│   └── feedback_catalog.json                   # 反馈模式目录（7,957 个探测）
│
├── augmentation/                                   # Phase 2 产出
│   ├── source_analysis.md                      # TextWorld/ALFWorld 源码分析
│   ├── augmentation_plan.json                  # 12 条增强规则定义
│   ├── augmented_env.py                        # AugmentedAlfWorldEnv Wrapper（558 行）
│   ├── smoke_test.py                           # 冒烟测试脚本
│   └── smoke_test_result.json                  # 冒烟测试结果（14/14 通过）
│
└── verification/                               # Phase 3 产出
    ├── verify_runner.py                        # 验证脚本
    ├── verify_results.json                     # 验证数据
    └── verify_report.md                        # 完整验证报告
```
