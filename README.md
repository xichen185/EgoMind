# EgoMind: 通用具身智能机器人模型框架

## 概述

**EgoMind** 是一个模块化的通用多任务具身智能机器人框架，整合了多模态感知、任务规划、动作执行、记忆系统和反馈学习，形成完整的 **感知-认知-执行-反馈** 闭环架构。

本框架参考了当前具身智能领域的前沿研究，包括 VLA（视觉-语言-动作）模型、大语言模型驱动的任务规划、以及强化学习与模仿学习的结合。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        交互层 (Interaction Layer)                    │
│         自然语言指令 │ 语音交互 │ 手势识别 │ 人类反馈                │
├─────────────────────────────────────────────────────────────────────┤
│                        认知层 (Cognition Layer)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  任务规划器   │  │   世界模型   │  │   知识图谱   │              │
│  │ TaskPlanner  │  │  WorldModel  │  │   (扩展)     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                        感知层 (Perception Layer)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐           │
│  │  视觉    │ │  触觉    │ │  本体感觉 │ │   感知融合   │           │
│  │ Camera   │ │ Tactile  │ │Proprioception│  Fusion    │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘           │
├─────────────────────────────────────────────────────────────────────┤
│                        执行层 (Action Layer)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  导航控制器   │  │  操作控制器   │  │   技能库     │              │
│  │ Navigation   │  │Manipulation │  │ SkillLibrary │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                        记忆层 (Memory Layer)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  情景记忆     │  │  语义记忆     │  │   工作记忆   │              │
│  │   Episodic   │  │   Semantic   │  │   Working    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                        反馈层 (Feedback Layer)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   奖励模型    │  │  学习优化器   │  │   纠正学习   │              │
│  │ RewardModel  │  │  Optimizer   │  │  Correction  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    核心智能体 (EmbodiedAgent)                        │
│              状态管理 │ 模块协调 │ 生命周期管理 │ 回调系统            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. 核心智能体 (`core/agent.py`)

`EmbodiedAgent` 是框架的中央控制器，负责：
- 管理智能体生命周期（启动/停止）
- 协调各模块间的数据流
- 维护执行状态机
- 提供任务执行入口

```python
from core.agent import EmbodiedAgent, AgentConfig

config = AgentConfig(
    perception_interval=0.1,
    enable_learning=True,
    memory_capacity=10000
)
agent = EmbodiedAgent(agent_id="robot_001", config=config)
agent.register_modules(perception=..., task_planner=..., ...)
agent.start()
result = agent.execute_task("去厨房拿杯子")
```

### 2. 多模态感知 (`perception/multimodal_perception.py`)

`MultimodalPerception` 整合多种传感器输入：
- **CameraSensor**: 视觉输入（RGB-D图像、物体检测）
- **MicrophoneSensor**: 语音/语言输入
- **TactileSensor**: 触觉/力觉输入
- **ProprioceptionSensor**: 本体感觉（关节状态、末端执行器位姿）

`PerceptionFusion` 将多模态数据融合为统一的环境表示。

### 3. 任务规划 (`cognition/task_planner.py`)

`TaskPlanner` 提供分层规划能力：
- **LLMPlanner**: 基于大语言模型的高层任务分解
- **SymbolicPlanner**: 基于符号表示的低层路径规划
- **Replanning**: 执行失败时的动态重新规划

### 4. 世界模型 (`cognition/world_model.py`)

`WorldModel` 维护环境的结构化内部表示：
- 物体状态跟踪（位置、类别、属性）
- 空间地图（占用网格）
- 场景关系图
- 状态预测

### 5. 动作执行 (`action/action_executor.py`)

`ActionExecutor` 管理底层控制器：
- **NavigationController**: 移动基座控制
- **ManipulationController**: 机械臂/夹爪控制
- **InteractionController**: 语音/显示交互

### 6. 技能库 (`action/skill_library.py`)

`SkillLibrary` 管理可复用的原子技能和复合技能：
- 技能定义与检索
- 从示教学习新技能
- 基于反馈更新技能成功率
- 技能组合与泛化

### 7. 记忆系统 (`memory/episodic_memory.py`)

`EpisodicMemory` 实现多层级记忆：
- **情景记忆**: 存储完整经验片段
- **语义记忆**: 提取概念性知识
- **短期缓冲**: 临时工作记忆
- **相似检索**: 基于内容的经验检索

### 8. 反馈循环 (`feedback/feedback_loop.py`)

`FeedbackLoop` 驱动持续学习：
- **RewardModel**: 自动奖励计算
- **LearningOptimizer**: 策略参数优化
- **CorrectionHandler**: 人类纠正处理

---

## 快速开始

### 安装

```bash
# 克隆或下载本框架
cd embodied_agent

# 安装依赖（仅需numpy）
pip install numpy
```

### 运行演示

```bash
python demo.py
```

### 基本使用

```python
from core.agent import EmbodiedAgent, AgentConfig
from perception.multimodal_perception import MultimodalPerception, CameraSensor
from cognition.task_planner import TaskPlanner
from cognition.world_model import WorldModel
from action.action_executor import ActionExecutor
from action.skill_library import SkillLibrary
from memory.episodic_memory import EpisodicMemory
from feedback.feedback_loop import FeedbackLoop

# 1. 创建智能体
agent = EmbodiedAgent("my_robot", AgentConfig())

# 2. 创建并注册模块
perception = MultimodalPerception()
perception.register_sensor("camera", CameraSensor())

agent.register_modules(
    perception=perception,
    task_planner=TaskPlanner(),
    world_model=WorldModel(),
    action_executor=ActionExecutor(),
    skill_library=SkillLibrary(),
    memory=EpisodicMemory(),
    feedback_loop=FeedbackLoop()
)

# 3. 启动并执行任务
agent.start()
result = agent.execute_task("去客厅拿遥控器")
print(f"任务结果: {result['status']}")
agent.stop()
```

---

## 扩展指南

### 添加自定义传感器

```python
from perception.multimodal_perception import BaseSensor, SensorData

class MySensor(BaseSensor):
    def read(self) -> SensorData:
        return SensorData(
            modality="my_modality",
            timestamp=time.time(),
            data={"value": 42}
        )
    
    def is_available(self) -> bool:
        return True

perception.register_sensor("my_sensor", MySensor())
```

### 添加自定义技能

```python
from action.skill_library import Skill

skill_library.add_skill(Skill(
    name="my_skill",
    description="我的自定义技能",
    skill_type="atomic",
    preconditions=["ready"],
    effects=["done"]
))
```

### 接入真实LLM

修改 `cognition/task_planner.py` 中的 `LLMPlanner.plan()` 方法：

```python
def plan(self, instruction, observation, context):
    # 替换为真实API调用
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"规划任务: {instruction}"}]
    )
    return self.parse_plan(response.choices[0].message.content)
```

---

## 技术特点

| 特性 | 说明 |
|------|------|
| **模块化设计** | 各组件可独立替换和扩展 |
| **多模态融合** | 支持视觉、语言、触觉、本体感觉的统一表示 |
| **分层规划** | LLM高层规划 + 符号低层规划的混合架构 |
| **持续学习** | 基于反馈的在线技能学习和策略优化 |
| **记忆系统** | 情景记忆 + 语义记忆的双系统架构 |
| **闭环反馈** | 执行-观察-评估-学习的完整循环 |

---

## 应用场景

- **服务机器人**: 家庭助理、导览机器人
- **工业机器人**: 柔性制造、协作装配
- **研究平台**: 具身智能算法验证
- **教育培训**: 机器人学教学演示

---

## 参考文献

1. Jiang et al. "Embodied Intelligence: The Key to Unblocking Generalized Artificial Intelligence." arXiv:2505.06897, 2025.
2. "AIRSHIP: Empowering General-Purpose Intelligent Robots through Open-Source Embodied AI." TechRxiv, 2025.
3. "A Survey: Learning Embodied Intelligence from Physical Simulators and World Models." arXiv:2507.00917, 2025.
4. 具身智能时代：机器人开发全栈技术图谱与实战指南（2026版）

---

## 许可证

MIT License

---

## 作者

xichen185

版本: 1.0.0
