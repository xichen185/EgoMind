"""
反馈模块：处理执行反馈，驱动持续学习与优化
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    REWARD = "reward"           # 数值奖励
    CORRECTION = "correction"   # 纠正指令
    DEMONSTRATION = "demonstration"  # 示范
    EVALUATION = "evaluation"   # 评价


@dataclass
class FeedbackSignal:
    """反馈信号"""
    episode_id: str
    feedback_type: FeedbackType
    content: Dict
    timestamp: float
    source: str = "human"  # human 或 environment


class RewardModel:
    """
    奖励模型：评估动作质量并生成奖励信号
    """

    def __init__(self):
        self.task_rewards = {}
        self.baseline_reward = 0.0

    def compute_reward(self, episode: Dict) -> float:
        """
        计算经验片段的奖励

        Args:
            episode: 经验数据

        Returns:
            奖励值
        """
        reward = 0.0

        # 任务成功奖励
        if episode.get("status") == "success":
            reward += 1.0
        else:
            reward -= 1.0

        # 效率奖励（时间越短越好）
        duration = episode.get("duration", 0.0)
        if duration > 0:
            reward += max(0, 1.0 - duration / 60.0) * 0.5

        # 步骤数惩罚
        num_steps = len(episode.get("steps", []))
        reward -= num_steps * 0.05

        return reward

    def update_baseline(self, rewards: List[float]):
        """更新奖励基线"""
        if rewards:
            self.baseline_reward = np.mean(rewards)


class LearningOptimizer:
    """
    学习优化器：基于反馈调整策略参数
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.parameter_updates = {}

    def optimize(self, feedback: FeedbackSignal, current_params: Dict) -> Dict:
        """
        基于反馈优化参数

        Args:
            feedback: 反馈信号
            current_params: 当前参数

        Returns:
            更新后的参数
        """
        updated_params = current_params.copy()

        if feedback.feedback_type == FeedbackType.REWARD:
            reward = feedback.content.get("value", 0.0)
            # 简单的梯度上升更新
            for key, value in current_params.items():
                if isinstance(value, (int, float)):
                    updated_params[key] = value + self.learning_rate * reward

        elif feedback.feedback_type == FeedbackType.CORRECTION:
            correction = feedback.content.get("correction", {})
            updated_params.update(correction)

        return updated_params


class FeedbackLoop:
    """
    反馈循环模块

    处理来自环境和人类的反馈，驱动智能体持续改进。
    """

    def __init__(self):
        self.reward_model = RewardModel()
        self.optimizer = LearningOptimizer()
        self.feedback_history: List[FeedbackSignal] = []
        self.correction_handlers: List[Callable] = []

    def process(self, episode_id: str, feedback: Dict):
        """
        处理反馈

        Args:
            episode_id: 经验片段ID
            feedback: 反馈内容
        """
        feedback_type = FeedbackType(feedback.get("type", "reward"))

        signal = FeedbackSignal(
            episode_id=episode_id,
            feedback_type=feedback_type,
            content=feedback,
            timestamp=time.time()
        )

        self.feedback_history.append(signal)
        logger.info(f"处理反馈: {episode_id} - {feedback_type.value}")

        # 根据反馈类型处理
        if feedback_type == FeedbackType.REWARD:
            self._process_reward(signal)
        elif feedback_type == FeedbackType.CORRECTION:
            self._process_correction(signal)
        elif feedback_type == FeedbackType.DEMONSTRATION:
            self._process_demonstration(signal)

    def _process_reward(self, signal: FeedbackSignal):
        """处理奖励反馈"""
        reward_value = signal.content.get("value", 0.0)
        logger.info(f"奖励信号: {reward_value}")

        # 更新奖励模型
        self.reward_model.update_baseline([reward_value])

    def _process_correction(self, signal: FeedbackSignal):
        """处理纠正反馈"""
        correction = signal.content.get("correction", {})
        logger.info(f"纠正反馈: {correction}")

        # 触发纠正处理回调
        for handler in self.correction_handlers:
            try:
                handler(signal.episode_id, correction)
            except Exception as e:
                logger.error(f"纠正处理错误: {e}")

    def _process_demonstration(self, signal: FeedbackSignal):
        """处理示范反馈"""
        demonstration = signal.content.get("trajectory", [])
        logger.info(f"示范反馈: {len(demonstration)} 步")

    def register_correction_handler(self, handler: Callable):
        """注册纠正处理回调"""
        self.correction_handlers.append(handler)

    def get_feedback_summary(self, episode_id: Optional[str] = None) -> Dict:
        """
        获取反馈摘要

        Args:
            episode_id: 可选的特定经验片段ID

        Returns:
            反馈统计
        """
        feedbacks = self.feedback_history
        if episode_id:
            feedbacks = [f for f in feedbacks if f.episode_id == episode_id]

        if not feedbacks:
            return {"count": 0}

        type_counts = {}
        for f in feedbacks:
            type_counts[f.feedback_type.value] = type_counts.get(f.feedback_type.value, 0) + 1

        return {
            "count": len(feedbacks),
            "type_distribution": type_counts,
            "human_feedback_ratio": sum(1 for f in feedbacks if f.source == "human") / len(feedbacks)
        }

    def compute_advantage(self, episode: Dict) -> float:
        """
        计算优势函数（用于强化学习）

        Args:
            episode: 经验数据

        Returns:
            优势值
        """
        reward = self.reward_model.compute_reward(episode)
        return reward - self.reward_model.baseline_reward


import time