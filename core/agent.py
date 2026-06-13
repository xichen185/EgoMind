"""
核心智能体模块：整合感知、认知、执行、记忆与反馈的中央控制器
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """智能体运行状态"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    ERROR = "error"


@dataclass
class AgentConfig:
    """智能体配置参数"""
    perception_interval: float = 0.1  # 感知周期（秒）
    planning_timeout: float = 30.0    # 规划超时（秒）
    execution_timeout: float = 60.0   # 执行超时（秒）
    enable_learning: bool = True      # 是否启用持续学习
    memory_capacity: int = 10000      # 记忆容量
    feedback_threshold: float = 0.5   # 反馈置信度阈值
    debug_mode: bool = False          # 调试模式


class EmbodiedAgent:
    """
    通用具身智能体主类

    整合多模态感知、任务规划、动作执行、记忆系统和反馈学习，
    形成完整的感知-认知-执行-反馈闭环。
    """

    def __init__(self, agent_id: str, config: Optional[AgentConfig] = None):
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        self.state = AgentState.IDLE
        self._running = False
        self._lock = threading.RLock()

        # 核心模块（将在初始化时注入）
        self.perception = None
        self.task_planner = None
        self.world_model = None
        self.action_executor = None
        self.skill_library = None
        self.memory = None
        self.feedback_loop = None

        # 回调函数
        self._state_callbacks: List[Callable[[AgentState, AgentState], None]] = []
        self._task_callbacks: List[Callable[[str, Any], None]] = []

        logger.info(f"智能体 {agent_id} 已初始化")

    def register_modules(self, **modules):
        """
        注册核心模块

        Args:
            perception: 多模态感知模块
            task_planner: 任务规划模块
            world_model: 世界模型模块
            action_executor: 动作执行模块
            skill_library: 技能库模块
            memory: 记忆模块
            feedback_loop: 反馈模块
        """
        valid_modules = {
            'perception', 'task_planner', 'world_model',
            'action_executor', 'skill_library', 'memory', 'feedback_loop'
        }
        for name, module in modules.items():
            if name in valid_modules:
                setattr(self, name, module)
                logger.info(f"模块 {name} 已注册")
            else:
                logger.warning(f"未知模块: {name}")

    def on_state_change(self, callback: Callable[[AgentState, AgentState], None]):
        """注册状态变化回调"""
        self._state_callbacks.append(callback)

    def on_task_complete(self, callback: Callable[[str, Any], None]):
        """注册任务完成回调"""
        self._task_callbacks.append(callback)

    def _set_state(self, new_state: AgentState):
        """设置新状态并触发回调"""
        with self._lock:
            old_state = self.state
            if old_state != new_state:
                self.state = new_state
                logger.info(f"状态变化: {old_state.value} -> {new_state.value}")
                for cb in self._state_callbacks:
                    try:
                        cb(old_state, new_state)
                    except Exception as e:
                        logger.error(f"状态回调错误: {e}")

    def start(self):
        """启动智能体主循环"""
        if self._running:
            logger.warning("智能体已在运行中")
            return

        self._running = True
        logger.info(f"智能体 {self.agent_id} 启动")

        # 启动后台感知线程
        self._perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self._perception_thread.start()

    def stop(self):
        """停止智能体"""
        self._running = False
        self._set_state(AgentState.IDLE)
        logger.info(f"智能体 {self.agent_id} 已停止")

    def _perception_loop(self):
        """后台感知循环"""
        while self._running:
            if self.perception and self.state not in [AgentState.ERROR]:
                try:
                    observation = self.perception.observe()
                    if self.world_model:
                        self.world_model.update(observation)
                    if self.memory:
                        self.memory.store_observation(observation)
                except Exception as e:
                    logger.error(f"感知循环错误: {e}")
            time.sleep(self.config.perception_interval)

    def execute_task(self, instruction: str, context: Optional[Dict] = None) -> Dict:
        """
        执行用户指令

        Args:
            instruction: 自然语言指令
            context: 额外上下文信息

        Returns:
            执行结果字典
        """
        start_time = time.time()
        result = {
            "task_id": f"task_{int(start_time * 1000)}",
            "instruction": instruction,
            "status": "pending",
            "steps": [],
            "duration": 0.0,
            "error": None
        }

        try:
            # 1. 感知当前环境
            self._set_state(AgentState.PERCEIVING)
            observation = self.perception.observe() if self.perception else {}

            # 2. 任务规划
            self._set_state(AgentState.PLANNING)
            plan = self.task_planner.plan(instruction, observation, context) if self.task_planner else []
            result["plan"] = plan

            # 3. 执行动作序列
            self._set_state(AgentState.EXECUTING)
            for step in plan:
                step_result = self._execute_step(step, observation)
                result["steps"].append(step_result)

                if not step_result.get("success", False):
                    raise RuntimeError(f"步骤执行失败: {step}")

                # 更新观察
                observation = self.perception.observe() if self.perception else {}

            result["status"] = "success"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            self._set_state(AgentState.ERROR)
            logger.error(f"任务执行失败: {e}")

        finally:
            result["duration"] = time.time() - start_time
            self._set_state(AgentState.IDLE)

            # 存储经验
            if self.memory and self.config.enable_learning:
                self.memory.store_episode(result)

            # 触发回调
            for cb in self._task_callbacks:
                try:
                    cb(instruction, result)
                except Exception as e:
                    logger.error(f"任务回调错误: {e}")

        return result

    def _execute_step(self, step: Dict, observation: Dict) -> Dict:
        """执行单个步骤"""
        step_type = step.get("type", "unknown")
        step_result = {
            "type": step_type,
            "success": False,
            "details": {}
        }

        if step_type == "navigate" and self.action_executor:
            step_result["details"] = self.action_executor.navigate(step.get("target"), observation)
            step_result["success"] = step_result["details"].get("success", False)

        elif step_type == "manipulate" and self.action_executor:
            step_result["details"] = self.action_executor.manipulate(
                step.get("action"), step.get("target"), observation
            )
            step_result["success"] = step_result["details"].get("success", False)

        elif step_type == "interact" and self.action_executor:
            step_result["details"] = self.action_executor.interact(step.get("action"), step.get("target"))
            step_result["success"] = step_result["details"].get("success", False)

        elif step_type == "learn" and self.skill_library:
            step_result["details"] = self.skill_library.learn_skill(step.get("skill_name"), step.get("demonstration"))
            step_result["success"] = True

        else:
            step_result["details"] = {"message": f"未识别的步骤类型: {step_type}"}

        return step_result

    def learn_from_feedback(self, episode_id: str, feedback: Dict):
        """
        基于反馈进行学习

        Args:
            episode_id: 经验片段ID
            feedback: 反馈信息（如奖励、纠正等）
        """
        if not self.config.enable_learning:
            return

        self._set_state(AgentState.LEARNING)
        try:
            if self.feedback_loop:
                self.feedback_loop.process(episode_id, feedback)
            if self.skill_library:
                self.skill_library.update_from_feedback(episode_id, feedback)
            logger.info(f"从反馈中学习完成: {episode_id}")
        except Exception as e:
            logger.error(f"学习过程错误: {e}")
        finally:
            self._set_state(AgentState.IDLE)

    def get_status(self) -> Dict:
        """获取智能体当前状态"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "running": self._running,
            "modules": {
                "perception": self.perception is not None,
                "task_planner": self.task_planner is not None,
                "world_model": self.world_model is not None,
                "action_executor": self.action_executor is not None,
                "skill_library": self.skill_library is not None,
                "memory": self.memory is not None,
                "feedback_loop": self.feedback_loop is not None,
            }
        }