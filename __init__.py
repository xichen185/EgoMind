"""
EgoMind: 通用具身智能机器人模型框架

一个模块化的通用多任务机器人框架，支持：
- 多模态感知融合（视觉、语言、触觉、本体感觉）
- 基于大模型的任务规划与决策
- 灵活的动作执行与技能管理
- 持续学习与记忆系统
- 闭环反馈与自适应优化

作者: AI Assistant
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .core.agent import EmbodiedAgent
from .perception.multimodal_perception import MultimodalPerception
from .cognition.task_planner import TaskPlanner
from .cognition.world_model import WorldModel
from .action.action_executor import ActionExecutor
from .action.skill_library import SkillLibrary
from .memory.episodic_memory import EpisodicMemory
from .feedback.feedback_loop import FeedbackLoop

__all__ = [
    "EmbodiedAgent",
    "MultimodalPerception",
    "TaskPlanner",
    "WorldModel",
    "ActionExecutor",
    "SkillLibrary",
    "EpisodicMemory",
    "FeedbackLoop",
]