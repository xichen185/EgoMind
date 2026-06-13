"""
技能库模块：管理机器人可执行的原子技能和复合技能
"""

import json
import pickle
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """技能定义"""
    name: str
    description: str
    skill_type: str  # "atomic" 或 "composite"
    parameters: Dict = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    demonstration: List[Dict] = field(default_factory=list)
    success_rate: float = 0.0
    execution_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "success_rate": self.success_rate,
            "execution_count": self.execution_count
        }


class SkillTemplate(ABC):
    """技能模板基类"""

    @abstractmethod
    def execute(self, params: Dict, observation: Dict) -> Dict:
        """执行技能"""
        pass

    @abstractmethod
    def get_preconditions(self) -> List[str]:
        """获取前置条件"""
        pass


class GraspSkill(SkillTemplate):
    """抓取技能模板"""

    def execute(self, params: Dict, observation: Dict) -> Dict:
        target = params.get("target", "")
        logger.info(f"执行抓取技能: {target}")
        return {"success": True, "action": "grasp", "target": target}

    def get_preconditions(self) -> List[str]:
        return ["gripper_empty", "object_reachable"]


class PlaceSkill(SkillTemplate):
    """放置技能模板"""

    def execute(self, params: Dict, observation: Dict) -> Dict:
        target = params.get("target", "")
        logger.info(f"执行放置技能: {target}")
        return {"success": True, "action": "place", "target": target}

    def get_preconditions(self) -> List[str]:
        return ["holding_object"]


class NavigateSkill(SkillTemplate):
    """导航技能模板"""

    def execute(self, params: Dict, observation: Dict) -> Dict:
        target = params.get("target", "")
        logger.info(f"执行导航技能: {target}")
        return {"success": True, "action": "navigate", "target": target}

    def get_preconditions(self) -> List[str]:
        return ["map_available"]


class SkillLibrary:
    """
    技能库

    管理机器人的所有技能，支持技能学习、检索和更新。
    """

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.templates: Dict[str, SkillTemplate] = {
            "grasp": GraspSkill(),
            "place": PlaceSkill(),
            "navigate": NavigateSkill()
        }
        self.skill_embeddings: Dict[str, List[float]] = {}

    def add_skill(self, skill: Skill):
        """添加技能"""
        self.skills[skill.name] = skill
        logger.info(f"技能已添加: {skill.name}")

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有技能名称"""
        return list(self.skills.keys())

    def find_skills_by_type(self, skill_type: str) -> List[Skill]:
        """按类型查找技能"""
        return [s for s in self.skills.values() if s.skill_type == skill_type]

    def find_skills_for_task(self, task_description: str) -> List[Skill]:
        """
        根据任务描述查找相关技能

        Args:
            task_description: 任务描述

        Returns:
            相关技能列表
        """
        # 简化：基于关键词匹配
        keywords = task_description.lower().split()
        matched = []
        for skill in self.skills.values():
            skill_text = f"{skill.name} {skill.description}".lower()
            if any(kw in skill_text for kw in keywords):
                matched.append(skill)
        return matched

    def learn_skill(self, skill_name: str, demonstration: List[Dict]) -> Skill:
        """
        从示教中学习新技能

        Args:
            skill_name: 技能名称
            demonstration: 示教轨迹

        Returns:
            学习到的技能
        """
        logger.info(f"从示教中学习技能: {skill_name}")

        # 分析示教数据提取参数和条件
        skill = Skill(
            name=skill_name,
            description=f"从示教学习的技能: {skill_name}",
            skill_type="composite",
            demonstration=demonstration,
            success_rate=0.5  # 初始成功率
        )

        self.add_skill(skill)
        return skill

    def update_from_feedback(self, episode_id: str, feedback: Dict):
        """
        基于反馈更新技能

        Args:
            episode_id: 经验片段ID
            feedback: 反馈信息
        """
        success = feedback.get("success", False)
        skill_name = feedback.get("skill_name", "")

        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill.execution_count += 1

            # 更新成功率
            if success:
                skill.success_rate = (skill.success_rate * (skill.execution_count - 1) + 1.0) / skill.execution_count
            else:
                skill.success_rate = (skill.success_rate * (skill.execution_count - 1)) / skill.execution_count

            logger.info(f"技能 {skill_name} 成功率更新为: {skill.success_rate:.2f}")

    def compose_skills(self, skill_names: List[str], composite_name: str) -> Skill:
        """
        组合多个技能为复合技能

        Args:
            skill_names: 技能名称列表
            composite_name: 复合技能名称

        Returns:
            复合技能
        """
        steps = []
        for name in skill_names:
            skill = self.skills.get(name)
            if skill:
                steps.append(skill.to_dict())

        composite_skill = Skill(
            name=composite_name,
            description=f"复合技能: {' + '.join(skill_names)}",
            skill_type="composite",
            demonstration=steps
        )

        self.add_skill(composite_skill)
        return composite_skill

    def save(self, filepath: str):
        """保存技能库到文件"""
        data = {name: skill.to_dict() for name, skill in self.skills.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"技能库已保存到: {filepath}")

    def load(self, filepath: str):
        """从文件加载技能库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for name, skill_data in data.items():
            skill = Skill(
                name=skill_data["name"],
                description=skill_data["description"],
                skill_type=skill_data["skill_type"],
                parameters=skill_data.get("parameters", {}),
                preconditions=skill_data.get("preconditions", []),
                effects=skill_data.get("effects", []),
                success_rate=skill_data.get("success_rate", 0.0),
                execution_count=skill_data.get("execution_count", 0)
            )
            self.skills[name] = skill

        logger.info(f"技能库已从 {filepath} 加载，共 {len(self.skills)} 个技能")
