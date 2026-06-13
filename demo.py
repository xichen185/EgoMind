"""
EgoMind 框架演示脚本

展示如何组装和使用具身智能体框架的各个模块。
"""

import time
import numpy as np

from core.agent import EmbodiedAgent, AgentConfig
from perception.multimodal_perception import (
    MultimodalPerception, CameraSensor, MicrophoneSensor,
    TactileSensor, ProprioceptionSensor
)
from cognition.task_planner import TaskPlanner
from cognition.world_model import WorldModel
from action.action_executor import ActionExecutor
from action.skill_library import SkillLibrary, Skill
from memory.episodic_memory import EpisodicMemory
from feedback.feedback_loop import FeedbackLoop


def create_robot_agent(agent_id: str = "ego_001") -> EmbodiedAgent:
    """
    创建并配置一个完整的具身智能体
    """
    # 1. 创建智能体配置
    config = AgentConfig(
        perception_interval=0.5,
        planning_timeout=30.0,
        execution_timeout=60.0,
        enable_learning=True,
        memory_capacity=1000,
        debug_mode=True
    )

    # 2. 创建智能体
    agent = EmbodiedAgent(agent_id=agent_id, config=config)

    # 3. 创建并配置感知模块
    perception = MultimodalPerception()
    perception.register_sensor("camera", CameraSensor(resolution=(640, 480)))
    perception.register_sensor("microphone", MicrophoneSensor())
    perception.register_sensor("tactile", TactileSensor(num_sensors=10))
    perception.register_sensor("proprioception", ProprioceptionSensor(num_joints=7))

    # 4. 创建认知模块
    task_planner = TaskPlanner()
    world_model = WorldModel(map_size=(100, 100))

    # 5. 创建执行模块
    action_executor = ActionExecutor()
    skill_library = SkillLibrary()

    # 预置一些基础技能
    from action.skill_library import Skill
    skill_library.add_skill(Skill(
        name="pick_and_place",
        description="抓取并放置物体",
        skill_type="composite",
        preconditions=["gripper_empty", "object_reachable"],
        effects=["object_moved"]
    ))
    skill_library.add_skill(Skill(
        name="navigate_to",
        description="导航到指定位置",
        skill_type="atomic",
        preconditions=["map_available"],
        effects=["agent_at_target"]
    ))

    # 6. 创建记忆模块
    memory = EpisodicMemory(capacity=1000)

    # 7. 创建反馈模块
    feedback_loop = FeedbackLoop()

    # 8. 注册所有模块到智能体
    agent.register_modules(
        perception=perception,
        task_planner=task_planner,
        world_model=world_model,
        action_executor=action_executor,
        skill_library=skill_library,
        memory=memory,
        feedback_loop=feedback_loop
    )

    # 9. 注册回调
    def on_state_change(old_state, new_state):
        print(f"[状态] {old_state.value} -> {new_state.value}")

    def on_task_complete(instruction, result):
        print(f"[任务完成] '{instruction}' -> {result['status']} ({result['duration']:.2f}s)")

    agent.on_state_change(on_state_change)
    agent.on_task_complete(on_task_complete)

    return agent


def demo_basic_task():
    """演示基本任务执行"""
    print("=" * 60)
    print("演示 1: 基本任务执行")
    print("=" * 60)

    agent = create_robot_agent("demo_robot_01")
    agent.start()

    # 执行导航任务
    result = agent.execute_task("去厨房拿杯子")
    print(f"\n任务结果: {result['status']}")
    print(f"执行步骤: {len(result['steps'])}")
    for step in result['steps']:
        print(f"  - {step['type']}: {'成功' if step['success'] else '失败'}")

    # 执行操作任务
    result2 = agent.execute_task("把杯子放到桌子上")
    print(f"\n任务结果: {result2['status']}")

    agent.stop()
    print()


def demo_perception():
    """演示多模态感知"""
    print("=" * 60)
    print("演示 2: 多模态感知")
    print("=" * 60)

    perception = MultimodalPerception()
    perception.register_sensor("camera", CameraSensor())
    perception.register_sensor("tactile", TactileSensor())
    perception.register_sensor("proprioception", ProprioceptionSensor())

    # 获取观察
    obs = perception.observe()
    print(f"观察置信度: {obs['confidence']:.2f}")
    print(f"可用模态: {list(obs['modalities'].keys())}")

    # 场景理解
    scene = obs['scene_understanding']
    print(f"检测到的物体: {scene.get('objects', [])}")
    print(f"机器人状态: {scene.get('agent_state', {})}")
    print()


def demo_world_model():
    """演示世界模型"""
    print("=" * 60)
    print("演示 3: 世界模型")
    print("=" * 60)

    world_model = WorldModel()

    # 模拟多次观察更新
    for i in range(3):
        mock_obs = {
            "modalities": {},
            "scene_understanding": {
                "objects": [
                    {"id": f"cup_{i}", "class": "cup", "position": [1.0, 2.0, 0.5]},
                    {"id": "table_1", "class": "table", "position": [2.0, 3.0, 0.0]}
                ]
            }
        }
        world_model.update(mock_obs)

    print(f"世界模型状态: {world_model.get_state_summary()}")
    print(f"物体数量: {len(world_model.objects)}")

    # 查询
    result = world_model.query("where is cup")
    print(f"查询结果: {result}")
    print()


def demo_skill_learning():
    """演示技能学习"""
    print("=" * 60)
    print("演示 4: 技能库与学习")
    print("=" * 60)

    skill_lib = SkillLibrary()

    # 添加技能
    skill_lib.add_skill(Skill(
        name="pour_water",
        description="倒水",
        skill_type="composite"
    ))

    # 从示教学习
    demonstration = [
        {"action": "grasp", "target": "bottle"},
        {"action": "pour", "target": "cup"},
        {"action": "place", "target": "table"}
    ]
    learned_skill = skill_lib.learn_skill("pour_water_v2", demonstration)
    print(f"学习到的技能: {learned_skill.name}")
    print(f"技能列表: {skill_lib.list_skills()}")

    # 基于反馈更新
    skill_lib.update_from_feedback("ep_001", {
        "skill_name": "pour_water",
        "success": True
    })
    print(f"更新后成功率: {skill_lib.get_skill('pour_water').success_rate}")
    print()


def demo_memory():
    """演示记忆系统"""
    print("=" * 60)
    print("演示 5: 记忆系统")
    print("=" * 60)

    memory = EpisodicMemory(capacity=100)

    # 存储经验
    for i in range(5):
        episode = {
            "task_id": f"task_{i}",
            "instruction": f"抓取物体 {i}",
            "status": "success" if i % 2 == 0 else "failed",
            "duration": 5.0 + i,
            "steps": [{"type": "grasp", "success": i % 2 == 0}]
        }
        memory.store_episode(episode)

    # 检索相似经验
    similar = memory.retrieve_similar_episodes("抓取物体", k=3)
    print(f"相似经验数量: {len(similar)}")

    # 统计
    stats = memory.get_statistics()
    print(f"记忆统计: {stats}")
    print()


def demo_feedback():
    """演示反馈循环"""
    print("=" * 60)
    print("演示 6: 反馈与学习优化")
    print("=" * 60)

    feedback = FeedbackLoop()

    # 处理奖励反馈
    feedback.process("task_001", {
        "type": "reward",
        "value": 1.0,
        "reason": "任务成功完成"
    })

    # 处理纠正反馈
    feedback.process("task_002", {
        "type": "correction",
        "correction": {"grasp_force": 0.8}
    })

    # 摘要
    summary = feedback.get_feedback_summary()
    print(f"反馈摘要: {summary}")
    print()


def demo_full_pipeline():
    """演示完整流水线"""
    print("=" * 60)
    print("演示 7: 完整执行流水线")
    print("=" * 60)

    agent = create_robot_agent("full_pipeline_robot")
    agent.start()

    # 执行复合任务
    instruction = "把书从书架拿到桌子上"
    print(f"指令: {instruction}")

    result = agent.execute_task(instruction)

    print(f"\n执行结果:")
    print(f"  状态: {result['status']}")
    print(f"  耗时: {result['duration']:.2f}秒")
    print(f"  步骤数: {len(result['steps'])}")

    if result['plan']:
        print(f"\n生成的计划:")
        for step in result['plan']:
            print(f"  - {step.get('description', step.get('type', 'unknown'))}")

    # 基于反馈学习
    agent.learn_from_feedback(result['task_id'], {
        "type": "reward",
        "value": 1.0 if result['status'] == 'success' else -1.0
    })

    # 查看记忆
    print(f"\n记忆统计: {agent.memory.get_statistics()}")

    agent.stop()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "EgoMind 具身智能框架演示" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demo_perception()
    demo_world_model()
    demo_skill_learning()
    demo_memory()
    demo_feedback()
    demo_basic_task()
    demo_full_pipeline()

    print("=" * 60)
    print("所有演示完成！")
    print("=" * 60)