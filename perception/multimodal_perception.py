"""
多模态感知模块：整合视觉、语言、触觉、本体感觉等多种传感器输入
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """传感器数据基类"""
    modality: str
    timestamp: float
    data: Any
    confidence: float = 1.0


@dataclass
class VisualObservation:
    """视觉观察结果"""
    image: np.ndarray
    depth_map: Optional[np.ndarray] = None
    detected_objects: List[Dict] = None
    scene_description: str = ""


@dataclass
class LanguageObservation:
    """语言观察结果"""
    text: str
    intent: Optional[str] = None
    entities: List[str] = None


@dataclass
class TactileObservation:
    """触觉观察结果"""
    force_vector: np.ndarray
    contact_points: List[Tuple[float, float, float]] = None
    texture: Optional[str] = None


@dataclass
class ProprioceptionObservation:
    """本体感觉观察结果"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: np.ndarray
    body_state: Dict = None


class BaseSensor(ABC):
    """传感器基类"""

    @abstractmethod
    def read(self) -> SensorData:
        """读取传感器数据"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查传感器是否可用"""
        pass


class CameraSensor(BaseSensor):
    """摄像头传感器（模拟）"""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.resolution = resolution
        self._available = True

    def read(self) -> SensorData:
        # 模拟生成随机图像数据
        image = np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8)
        depth = np.random.rand(*self.resolution).astype(np.float32)
        return SensorData(
            modality="visual",
            timestamp=0.0,
            data=VisualObservation(image=image, depth_map=depth),
            confidence=0.95
        )

    def is_available(self) -> bool:
        return self._available


class MicrophoneSensor(BaseSensor):
    """麦克风传感器（模拟）"""

    def __init__(self):
        self._available = True

    def read(self) -> SensorData:
        return SensorData(
            modality="audio",
            timestamp=0.0,
            data=LanguageObservation(
                text="模拟语音输入",
                intent="query",
                entities=[]
            ),
            confidence=0.9
        )

    def is_available(self) -> bool:
        return self._available


class TactileSensor(BaseSensor):
    """触觉传感器（模拟）"""

    def __init__(self, num_sensors: int = 10):
        self.num_sensors = num_sensors
        self._available = True

    def read(self) -> SensorData:
        force = np.random.rand(3) * 10.0
        contacts = [(np.random.rand(), np.random.rand(), np.random.rand())
                    for _ in range(self.num_sensors)]
        return SensorData(
            modality="tactile",
            timestamp=0.0,
            data=TactileObservation(
                force_vector=force,
                contact_points=contacts,
                texture="smooth"
            ),
            confidence=0.85
        )

    def is_available(self) -> bool:
        return self._available


class ProprioceptionSensor(BaseSensor):
    """本体感觉传感器（模拟）"""

    def __init__(self, num_joints: int = 7):
        self.num_joints = num_joints
        self._available = True

    def read(self) -> SensorData:
        positions = np.random.rand(self.num_joints) * 2 * np.pi
        velocities = np.random.randn(self.num_joints) * 0.1
        pose = np.eye(4)
        pose[:3, 3] = np.random.rand(3)
        return SensorData(
            modality="proprioception",
            timestamp=0.0,
            data=ProprioceptionObservation(
                joint_positions=positions,
                joint_velocities=velocities,
                end_effector_pose=pose,
                body_state={"temperature": 36.5, "battery": 0.85}
            ),
            confidence=0.99
        )

    def is_available(self) -> bool:
        return self._available


class PerceptionFusion:
    """感知融合器：将多模态数据融合为统一的环境表示"""

    def __init__(self):
        self.fusion_weights = {
            "visual": 0.4,
            "audio": 0.2,
            "tactile": 0.2,
            "proprioception": 0.2
        }

    def fuse(self, sensor_data_list: List[SensorData]) -> Dict:
        """
        融合多模态传感器数据

        Args:
            sensor_data_list: 传感器数据列表

        Returns:
            融合后的环境表示字典
        """
        fused = {
            "timestamp": 0.0,
            "modalities": {},
            "scene_understanding": {},
            "confidence": 0.0
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for data in sensor_data_list:
            modality = data.modality
            fused["modalities"][modality] = data.data
            weight = self.fusion_weights.get(modality, 0.1)
            weighted_confidence += data.confidence * weight
            total_weight += weight

        if total_weight > 0:
            fused["confidence"] = weighted_confidence / total_weight

        # 场景理解（简化版）
        fused["scene_understanding"] = self._scene_understanding(fused["modalities"])

        return fused

    def _scene_understanding(self, modalities: Dict) -> Dict:
        """基于多模态数据进行场景理解"""
        understanding = {
            "objects": [],
            "spatial_relations": {},
            "agent_state": {},
            "human_presence": False
        }

        # 从视觉模态提取物体信息
        if "visual" in modalities:
            visual = modalities["visual"]
            if isinstance(visual, VisualObservation) and visual.detected_objects:
                understanding["objects"] = visual.detected_objects

        # 从本体感觉提取状态
        if "proprioception" in modalities:
            proprio = modalities["proprioception"]
            if isinstance(proprio, ProprioceptionObservation):
                understanding["agent_state"] = {
                    "joint_positions": proprio.joint_positions.tolist(),
                    "end_effector_position": proprio.end_effector_pose[:3, 3].tolist()
                }

        # 从音频检测人类存在
        if "audio" in modalities:
            audio = modalities["audio"]
            if isinstance(audio, LanguageObservation) and audio.text:
                understanding["human_presence"] = True

        return understanding


class MultimodalPerception:
    """
    多模态感知模块

    管理多种传感器，提供统一的数据获取和融合接口。
    """

    def __init__(self):
        self.sensors: Dict[str, BaseSensor] = {}
        self.fusion = PerceptionFusion()
        self._last_observation: Optional[Dict] = None

    def register_sensor(self, name: str, sensor: BaseSensor):
        """注册传感器"""
        self.sensors[name] = sensor
        logger.info(f"传感器 {name} ({sensor.__class__.__name__}) 已注册")

    def observe(self) -> Dict:
        """
        获取当前环境观察

        Returns:
            融合后的环境表示
        """
        sensor_data_list = []

        for name, sensor in self.sensors.items():
            if sensor.is_available():
                try:
                    data = sensor.read()
                    sensor_data_list.append(data)
                except Exception as e:
                    logger.error(f"传感器 {name} 读取失败: {e}")

        if not sensor_data_list:
            logger.warning("没有可用的传感器数据")
            return {"timestamp": 0.0, "modalities": {}, "scene_understanding": {}, "confidence": 0.0}

        fused_observation = self.fusion.fuse(sensor_data_list)
        self._last_observation = fused_observation
        return fused_observation

    def get_last_observation(self) -> Optional[Dict]:
        """获取上一次观察结果"""
        return self._last_observation

    def get_sensor_status(self) -> Dict[str, bool]:
        """获取所有传感器状态"""
        return {name: sensor.is_available() for name, sensor in self.sensors.items()}

    def calibrate(self):
        """校准所有传感器"""
        logger.info("开始传感器校准...")
        for name, sensor in self.sensors.items():
            logger.info(f"校准传感器: {name}")
        logger.info("传感器校准完成")
