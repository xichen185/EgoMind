"""
世界模型模块：维护机器人对环境的内部表示与预测能力
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """物体状态"""
    id: str
    class_name: str
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    properties: Dict = field(default_factory=dict)
    last_updated: float = 0.0


@dataclass
class SpatialMap:
    """空间地图"""
    occupancy_grid: np.ndarray
    resolution: float = 0.05  # 米/格
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    semantic_labels: Dict[Tuple[int, int], str] = field(default_factory=dict)


class WorldModel:
    """
    世界模型

    维护环境的结构化表示，包括：
    - 物体状态跟踪
    - 空间地图
    - 场景语义理解
    - 状态预测
    """

    def __init__(self, map_size: Tuple[int, int] = (100, 100)):
        self.objects: Dict[str, ObjectState] = {}
        self.spatial_map = SpatialMap(
            occupancy_grid=np.zeros(map_size, dtype=np.float32)
        )
        self.scene_graph: Dict = {"nodes": [], "edges": []}
        self.history = deque(maxlen=1000)
        self._time_step = 0

    def update(self, observation: Dict):
        """
        基于观察更新世界模型

        Args:
            observation: 多模态观察结果
        """
        self._time_step += 1
        self.history.append({
            "timestep": self._time_step,
            "observation": observation
        })

        # 更新空间地图
        self._update_spatial_map(observation)

        # 更新物体状态
        self._update_objects(observation)

        # 更新场景图
        self._update_scene_graph()

        logger.debug(f"世界模型已更新 (t={self._time_step})")

    def _update_spatial_map(self, observation: Dict):
        """更新空间地图"""
        modalities = observation.get("modalities", {})

        if "visual" in modalities:
            visual = modalities["visual"]
            if hasattr(visual, 'depth_map') and visual.depth_map is not None:
                # 简化：将深度信息整合到占用网格
                depth = visual.depth_map
                # 实际应用中应使用SLAM算法
                pass

    def _update_objects(self, observation: Dict):
        """更新物体状态"""
        scene = observation.get("scene_understanding", {})
        detected_objects = scene.get("objects", [])

        for obj_info in detected_objects:
            obj_id = obj_info.get("id", f"obj_{len(self.objects)}")
            if obj_id in self.objects:
                # 更新已有物体
                self.objects[obj_id].position = np.array(obj_info.get("position", [0, 0, 0]))
                self.objects[obj_id].last_updated = self._time_step
            else:
                # 添加新物体
                self.objects[obj_id] = ObjectState(
                    id=obj_id,
                    class_name=obj_info.get("class", "unknown"),
                    position=np.array(obj_info.get("position", [0, 0, 0])),
                    orientation=np.array([0, 0, 0, 1]),
                    velocity=np.zeros(3),
                    properties=obj_info.get("properties", {}),
                    last_updated=self._time_step
                )
                logger.info(f"发现新物体: {obj_id} ({obj_info.get('class', 'unknown')})")

    def _update_scene_graph(self):
        """更新场景关系图"""
        nodes = []
        edges = []

        for obj_id, obj_state in self.objects.items():
            nodes.append({
                "id": obj_id,
                "class": obj_state.class_name,
                "position": obj_state.position.tolist()
            })

        # 计算物体间空间关系
        obj_list = list(self.objects.values())
        for i, obj1 in enumerate(obj_list):
            for obj2 in obj_list[i+1:]:
                dist = np.linalg.norm(obj1.position - obj2.position)
                if dist < 1.0:  # 1米内认为相邻
                    edges.append({
                        "source": obj1.id,
                        "target": obj2.id,
                        "relation": "near",
                        "distance": float(dist)
                    })

        self.scene_graph = {"nodes": nodes, "edges": edges}

    def predict(self, action: Dict, steps: int = 1) -> Dict:
        """
        预测执行动作后的世界状态

        Args:
            action: 动作描述
            steps: 预测步数

        Returns:
            预测的状态
        """
        # 简化预测模型
        predicted_state = {
            "objects": {},
            "agent_position": np.zeros(3),
            "confidence": 0.7
        }

        action_type = action.get("type", "unknown")
        if action_type == "navigate":
            target = action.get("target", [0, 0, 0])
            predicted_state["agent_position"] = np.array(target)
        elif action_type == "grasp":
            target_obj = action.get("target", "")
            if target_obj in self.objects:
                predicted_state["objects"][target_obj] = {
                    "held_by": "agent",
                    "position": "unknown"
                }

        return predicted_state

    def get_object(self, obj_id: str) -> Optional[ObjectState]:
        """获取物体状态"""
        return self.objects.get(obj_id)

    def get_objects_by_class(self, class_name: str) -> List[ObjectState]:
        """按类别获取物体"""
        return [obj for obj in self.objects.values() if obj.class_name == class_name]

    def get_nearest_object(self, position: np.ndarray, class_name: Optional[str] = None) -> Optional[ObjectState]:
        """获取最近物体"""
        candidates = self.objects.values()
        if class_name:
            candidates = [obj for obj in candidates if obj.class_name == class_name]

        if not candidates:
            return None

        return min(candidates, key=lambda obj: np.linalg.norm(obj.position - position))

    def get_map(self) -> SpatialMap:
        """获取空间地图"""
        return self.spatial_map

    def query(self, query_str: str) -> Any:
        """
        查询世界模型

        Args:
            query_str: 查询字符串，如 "where is cup?", "what is near table?"

        Returns:
            查询结果
        """
        query_lower = query_str.lower()

        if "where is" in query_lower:
            obj_name = query_lower.replace("where is", "").strip().rstrip("?")
            objects = self.get_objects_by_class(obj_name)
            if objects:
                return objects[0].position.tolist()
            return None

        if "what is near" in query_lower:
            obj_name = query_lower.replace("what is near", "").strip().rstrip("?")
            # 查找相邻物体
            return [edge for edge in self.scene_graph["edges"]
                    if edge["source"] == obj_name or edge["target"] == obj_name]

        return None

    def get_state_summary(self) -> Dict:
        """获取世界状态摘要"""
        return {
            "num_objects": len(self.objects),
            "object_classes": list(set(obj.class_name for obj in self.objects.values())),
            "map_size": self.spatial_map.occupancy_grid.shape,
            "time_step": self._time_step,
            "scene_graph_nodes": len(self.scene_graph["nodes"]),
            "scene_graph_edges": len(self.scene_graph["edges"])
        }
