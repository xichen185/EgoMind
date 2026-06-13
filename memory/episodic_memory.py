"""
记忆模块：存储和管理机器人的经验、知识和学习历史
"""

import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """经验片段"""
    episode_id: str
    timestamp: float
    instruction: str
    observation: Dict = field(default_factory=dict)
    plan: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    outcome: Dict = field(default_factory=dict)
    reward: float = 0.0
    feedback: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SemanticMemory:
    """语义记忆项"""
    concept: str
    category: str
    attributes: Dict = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)  # (relation, target)


class EpisodicMemory:
    """
    情景记忆系统

    存储机器人的经验片段，支持检索、总结和泛化。
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self.semantic_memory: Dict[str, SemanticMemory] = {}
        self.short_term_buffer: List[Episode] = []
        self.buffer_size = 10

    def store_observation(self, observation: Dict):
        """存储观察（短期缓冲）"""
        self.short_term_buffer.append({
            "timestamp": time.time(),
            "observation": observation
        })
        if len(self.short_term_buffer) > self.buffer_size:
            self.short_term_buffer.pop(0)

    def store_episode(self, episode_data: Dict):
        """
        存储完整经验片段

        Args:
            episode_data: 经验数据
        """
        episode = Episode(
            episode_id=episode_data.get("task_id", f"ep_{time.time()}"),
            timestamp=time.time(),
            instruction=episode_data.get("instruction", ""),
            plan=episode_data.get("plan", []),
            actions=episode_data.get("steps", []),
            outcome={
                "status": episode_data.get("status", "unknown"),
                "duration": episode_data.get("duration", 0.0)
            },
            reward=1.0 if episode_data.get("status") == "success" else -1.0
        )

        self.episodes.append(episode)
        logger.info(f"经验片段已存储: {episode.episode_id}")

        # 提取语义知识
        self._extract_semantic_knowledge(episode)

    def _extract_semantic_knowledge(self, episode: Episode):
        """从经验中提取语义知识"""
        # 简单提取：记录任务类型与结果的关系
        instruction = episode.instruction.lower()

        if "抓取" in instruction or "拿" in instruction:
            concept = "grasp_task"
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = SemanticMemory(
                    concept=concept,
                    category="task_type",
                    attributes={"success_rate": 0.0, "count": 0}
                )

            mem = self.semantic_memory[concept]
            mem.attributes["count"] = mem.attributes.get("count", 0) + 1
            total = mem.attributes["count"]
            current_rate = mem.attributes.get("success_rate", 0.0)
            success = 1.0 if episode.outcome.get("status") == "success" else 0.0
            mem.attributes["success_rate"] = (current_rate * (total - 1) + success) / total

    def retrieve_similar_episodes(self, instruction: str, k: int = 5) -> List[Episode]:
        """
        检索相似经验

        Args:
            instruction: 查询指令
            k: 返回数量

        Returns:
            相似经验列表
        """
        # 简化：基于关键词相似度
        query_words = set(instruction.lower().split())
        scored = []

        for episode in self.episodes:
            ep_words = set(episode.instruction.lower().split())
            similarity = len(query_words & ep_words) / max(len(query_words), 1)
            scored.append((similarity, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def retrieve_successful_strategies(self, task_type: str) -> List[Dict]:
        """
        检索某类任务的成功策略

        Args:
            task_type: 任务类型

        Returns:
            成功策略列表
        """
        strategies = []
        for episode in self.episodes:
            if (task_type.lower() in episode.instruction.lower() and
                episode.outcome.get("status") == "success"):
                strategies.append({
                    "plan": episode.plan,
                    "actions": episode.actions,
                    "reward": episode.reward
                })
        return strategies

    def get_statistics(self) -> Dict:
        """获取记忆统计信息"""
        total = len(self.episodes)
        if total == 0:
            return {"total_episodes": 0}

        success_count = sum(1 for ep in self.episodes if ep.outcome.get("status") == "success")
        avg_reward = np.mean([ep.reward for ep in self.episodes])

        return {
            "total_episodes": total,
            "success_rate": success_count / total,
            "average_reward": avg_reward,
            "semantic_concepts": len(self.semantic_memory),
            "buffer_size": len(self.short_term_buffer)
        }

    def consolidate(self):
        """
        记忆巩固：将短期记忆整合到长期记忆
        """
        logger.info("执行记忆巩固...")
        # 清理冗余，提取模式
        self.short_term_buffer.clear()

    def save(self, filepath: str):
        """保存记忆到文件"""
        data = {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "semantic_memory": {k: asdict(v) for k, v in self.semantic_memory.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"记忆已保存到: {filepath}")

    def load(self, filepath: str):
        """从文件加载记忆"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ep_data in data.get("episodes", []):
            episode = Episode(**ep_data)
            self.episodes.append(episode)

        for concept, mem_data in data.get("semantic_memory", {}).items():
            self.semantic_memory[concept] = SemanticMemory(**mem_data)

        logger.info(f"记忆已从 {filepath} 加载")
