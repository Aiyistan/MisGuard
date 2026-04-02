"""
ConAttacker: 多智能体模拟框架，生成对抗性评论

合并了agents目录下所有智能体和simulators目录下所有生成逻辑
"""
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from config import AGENT_CONFIG, get_agent_config, get_prompts
from utils import get_logger, create_api_manager, VideoProcessor


# ==========================================
# 数据类
# ==========================================

@dataclass
class CommentContext:
    """评论上下文"""
    video_id: str
    video_description: str
    transcript: str
    label: str  # 'real' or 'fake'
    video_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    video_understanding: Optional[Dict[str, Any]] = None


@dataclass
class CommentNode:
    """评论节点"""
    comment_id: int
    content: str
    agent_role: str
    parent_id: Optional[int] = None
    depth: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    children: List['CommentNode'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "comment_id": self.comment_id,
            "parent_id": self.parent_id,
            "agent_role": self.agent_role,
            "content": self.content,
            "depth": self.depth,
            "timestamp": self.timestamp
        }


# ==========================================
# 基础智能体
# ==========================================

class BaseAgent:
    """基础智能体类"""

    def __init__(
        self,
        agent_type: str,
        stance: str = 'support',
        model: str = 'gpt-4o-mini',
        temperature: Optional[float] = None
    ):
        """
        初始化智能体

        Args:
            agent_type: 智能体类型
            stance: 立场 ('support' 或 'doubt')
            model: 使用的LLM模型
            temperature: 温度参数
        """
        self.agent_type = agent_type
        self.agent_role = agent_type
        self.stance = stance
        self.model = model

        # 加载配置
        self.config = get_agent_config(agent_type)
        self.name = self.config['name']
        self.role = self.config['role']
        self.temperature = temperature if temperature is not None else self.config['temperature']
        self.max_tokens = self.config['max_tokens']

        # 加载prompt模板
        self.prompts = get_prompts(agent_type)

        # API管理器
        self.api_manager = create_api_manager(model)

        # 日志
        self.logger = get_logger()

    def get_system_prompt(self, video_context: CommentContext) -> str:
        """获取系统prompt"""
        # 确定使用哪个prompt键
        if self.agent_type == 'skeptic':
            prompt_key = 'refuted_prompt' if self.stance == 'support' else 'doubt_prompt'
        else:
            prompt_key = f'{self.stance}_prompt'

        if prompt_key not in self.prompts:
            self.logger.error(f"未找到prompt: {prompt_key}")
            return ""

        system_template = self.prompts[prompt_key]['system']

        # 使用视频理解结果（如果有）
        if video_context.video_understanding:
            video_desc = video_context.video_understanding.get('full_description',
                                                               video_context.video_description)
        else:
            video_desc = video_context.video_description

        # 填充模板
        system_prompt = system_template.format(
            video_description=video_desc,
            transcript=video_context.transcript
        )

        return system_prompt

    def generate_main_comment(
        self,
        video_context: CommentContext,
        temperature: Optional[float] = None
    ) -> str:
        """生成主评论（根评论）"""
        # 获取prompt
        if self.agent_type == 'skeptic':
            prompt_key = 'refuted_prompt' if self.stance == 'support' else 'doubt_prompt'
        else:
            prompt_key = f'{self.stance}_prompt'

        if prompt_key not in self.prompts:
            self.logger.error(f"未找到prompt: {prompt_key}")
            return ""

        system_prompt = self.get_system_prompt(video_context)
        main_comment_prompt = self.prompts[prompt_key]['main_comment']

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": main_comment_prompt}
        ]

        # 调用API
        temp = temperature if temperature is not None else self.temperature
        response, usage = self.api_manager.call(
            messages,
            temperature=temp,
            max_tokens=self.max_tokens
        )

        self.logger.debug(f"{self.name} 生成主评论: {response[:50]}...")

        return response.strip()

    def generate_reply(
        self,
        video_context: CommentContext,
        target_comment: str,
        thread_history: List[str],
        temperature: Optional[float] = None
    ) -> str:
        """生成回复"""
        # 获取prompt
        if self.agent_type == 'skeptic':
            prompt_key = 'refuted_prompt' if self.stance == 'support' else 'doubt_prompt'
        else:
            prompt_key = f'{self.stance}_prompt'

        if prompt_key not in self.prompts:
            self.logger.error(f"未找到prompt: {prompt_key}")
            return ""

        system_prompt = self.get_system_prompt(video_context)

        # 构建对话历史
        if thread_history and len(thread_history) > 0:
            actual_history = thread_history[:-1] if len(thread_history) > 1 else []

            if actual_history:
                history_text = "\n".join([f"- {comment}" for comment in actual_history])
                history_text = f"之前的对话：\n{history_text}\n\n要回复的评论：{target_comment}"
            else:
                history_text = f"要回复的评论：{target_comment}"
        else:
            history_text = f"要回复的评论：{target_comment}"

        reply_prompt = self.prompts[prompt_key]['reply'].format(thread_history=history_text)

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reply_prompt}
        ]

        # 调用API
        temp = temperature if temperature is not None else self.temperature
        response, usage = self.api_manager.call(
            messages,
            temperature=temp,
            max_tokens=int(self.max_tokens * 0.6)
        )

        self.logger.debug(f"{self.name} 生成回复: {response[:50]}...")

        return response.strip()

    def __str__(self):
        return f"{self.name} ({self.agent_type}, stance={self.stance})"


# ==========================================
# 具体智能体类
# ==========================================

class Witness(BaseAgent):
    """目击者智能体"""
    def __init__(self, stance: str = 'support', model: str = 'gpt-4o-mini', **kwargs):
        super().__init__('witness', stance, model, **kwargs)


class Expert(BaseAgent):
    """领域专家智能体"""
    def __init__(self, stance: str = 'support', model: str = 'gpt-4o-mini', **kwargs):
        super().__init__('expert', stance, model, **kwargs)


class KnowledgeExpert(BaseAgent):
    """外部知识专家智能体"""
    def __init__(self, stance: str = 'support', model: str = 'gpt-4o-mini', **kwargs):
        super().__init__('knowledge_expert', stance, model, **kwargs)


class Skeptic(BaseAgent):
    """质疑者智能体"""
    def __init__(self, stance: str = 'support', model: str = 'gpt-4o-mini', **kwargs):
        super().__init__('skeptic', stance, model, **kwargs)


class Passerby(BaseAgent):
    """路人智能体"""
    def __init__(self, stance: str = 'support', model: str = 'gpt-4o-mini', **kwargs):
        super().__init__('passerby', stance, model, **kwargs)


# ==========================================
# 评论树
# ==========================================

class CommentTree:
    """评论树"""

    def __init__(self):
        self.nodes: Dict[int, CommentNode] = {}
        self.root_nodes: List[CommentNode] = []
        self.next_id = 1
        self.logger = get_logger()

    def add_main_comment(self, content: str, agent_role: str) -> CommentNode:
        """添加主评论（根节点）"""
        node = CommentNode(
            comment_id=self.next_id,
            content=content,
            agent_role=agent_role,
            parent_id=None,
            depth=0
        )

        self.nodes[self.next_id] = node
        self.root_nodes.append(node)
        self.next_id += 1

        self.logger.debug(f"添加主评论 #{node.comment_id} [{agent_role}]: {content[:30]}...")
        return node

    def add_reply(
        self,
        parent_id: int,
        content: str,
        agent_role: str
    ) -> Optional[CommentNode]:
        """添加回复"""
        if parent_id not in self.nodes:
            self.logger.error(f"父节点 {parent_id} 不存在")
            return None

        parent = self.nodes[parent_id]

        node = CommentNode(
            comment_id=self.next_id,
            content=content,
            agent_role=agent_role,
            parent_id=parent_id,
            depth=parent.depth + 1
        )

        self.nodes[self.next_id] = node
        parent.children.append(node)
        self.next_id += 1

        self.logger.debug(f"添加回复 #{node.comment_id} → #{parent_id} [{agent_role}]: {content[:30]}...")
        return node

    def get_thread(self, comment_id: int) -> List[str]:
        """获取评���链（从根到当前评论）"""
        if comment_id not in self.nodes:
            return []

        thread = []
        current = self.nodes[comment_id]

        # 回溯到根节点
        path = [current]
        while current.parent_id is not None:
            current = self.nodes[current.parent_id]
            path.append(current)

        # 反转（从根到当前）
        path.reverse()

        return [node.content for node in path]

    def get_all_comments(self) -> List[CommentNode]:
        """获取所有评论（按时间排序）"""
        return sorted(self.nodes.values(), key=lambda x: x.timestamp)

    def total_comments(self) -> int:
        """总评论数"""
        return len(self.nodes)

    def max_depth(self) -> int:
        """最大深度"""
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())

    def to_json(self) -> str:
        """转换为JSON字符串"""
        data = {
            "total_comments": self.total_comments(),
            "max_depth": self.max_depth(),
            "tree_structure": {
                "nodes": [node.to_dict() for node in self.get_all_comments()]
            }
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def save(self, output_path: str):
        """保存到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        self.logger.info(f"评论树已保存到: {output_path}")


# ==========================================
# 视频理解智能体
# ==========================================

class VideoUnderstandingAgent:
    """视频理解智能体"""

    def __init__(self, model: str = 'gpt-4o-mini', cache_dir: str = None):
        """
        初始化视频理解智能体

        Args:
            model: 使用的模型
            cache_dir: 缓存目录
        """
        self.model = model
        self.api_manager = create_api_manager(model)
        self.processor = VideoProcessor()
        self.logger = get_logger()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _get_cache_path(self, video_id: str) -> Path:
        """获取缓存文件路径"""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return self.cache_dir / f"{video_id}_understanding.json"
        return None

    def understand_video(
        self,
        video_context: CommentContext,
        num_frames: int = 8
    ) -> Dict[str, Any]:
        """理解视频内容"""
        self.logger.info(f"视频理解智能体正在分析视频: {video_context.video_id}")

        # 检查缓存
        cache_path = self._get_cache_path(video_context.video_id)
        if cache_path and cache_path.exists():
            self.logger.info(f"从缓存加载视频理解结果: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 获取视频帧路径
        if not video_context.video_path:
            self.logger.warning(f"CommentContext缺少video_path，使用文本模式")
            return self._understand_from_text(video_context)

        frame_paths = self.processor.load_frames_from_directory(
            video_context.video_path,
            num_frames=num_frames
        )

        if not frame_paths:
            self.logger.warning(f"无法获取视频帧，使用文本模式")
            return self._understand_from_text(video_context)

        # 构建多模态prompt
        messages = self._build_understanding_prompt(frame_paths, video_context)

        # 调用API
        try:
            response, usage = self.api_manager.call(
                messages,
                temperature=0.3,
                max_tokens=1000
            )
        except Exception as e:
            error_msg = str(e)
            if 'maximum' in error_msg.lower() or 'longer' in error_msg.lower():
                self.logger.warning(f"Prompt超长，尝试压缩图片重试...")
                messages = self._build_understanding_prompt(frame_paths, video_context, compress=True)
                response, usage = self.api_manager.call(messages, temperature=0.3, max_tokens=1000)
            else:
                raise

        # 解析响应
        understanding = self._parse_understanding(response)

        self.logger.info(f"视频理解完成: {understanding['event_summary'][:100]}...")

        # 保存到缓存
        if cache_path:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(understanding, f, ensure_ascii=False, indent=2)

        return understanding

    def _build_understanding_prompt(
        self,
        frame_paths: List[str],
        video_context: CommentContext,
        compress: bool = False
    ) -> List[Dict[str, Any]]:
        """构建视频理解的prompt"""
        system_prompt = """你是一个专业的视频内容分析专家。你的任务是详细分析这个视频的内容。重要原则：**语言匹配**：如果视频转写是中文，用中文；如果是英文，用英文

请仔细观察视频的每一帧，并提供以下信息：

1. **事件总结**（1-2句话）：视频展示了什么事件或场景？
2. **视觉细节**：详细描述你看到的视觉元素（人物、动作、表情、环境、物体等）
3. **关键时刻**：列出3-5个视频中的关键时刻或转折点
4. **人物和物体**：视频中有哪些主要人物和物体？
5. **地点和时间**：事件发生在什么地点？什么时间（如果能判断）？

请严格按照以下JSON格式输出：
```json
{
  "event_summary": "事件总结（1-2句话）",
  "visual_details": "详细的视觉描述（3-5句话）",
  "key_moments": ["关键时刻1", "关键时刻2", "关键时刻3"],
  "people_objects": "人物和物体描述",
  "location_time": "地点和时间"
}
```

重要：
- 基于视觉证据，不要编造
- 如果某些信息不确定，说"无法确定"
- 描述要具体、详细，便于其他AI理解"""

        # 构建用户消息
        user_content = []

        # 添加文本信息
        text_info = f"视频标题：{video_context.video_description}\n"
        if video_context.transcript:
            text_info += f"\n音频转写：\n{video_context.transcript}\n"

        user_content.append({
            "type": "text",
            "text": text_info + "\n请分析这个视频的内容："
        })

        # 添加帧图片
        for frame_path in frame_paths:
            base64_frame = self.processor.frame_to_base64(frame_path)
            if base64_frame:
                if compress:
                    base64_frame = self.processor.compress_image_base64(base64_frame)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_frame}"
                    }
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        return messages

    def _understand_from_text(self, video_context: CommentContext) -> Dict[str, Any]:
        """仅基于文本理解视频"""
        self.logger.info("使用文本模式理解视频")

        prompt = f"""基于以下文本信息，推断视频内容：

标题：{video_context.video_description}

转写：{video_context.transcript}

请推断这个视频可能展示的内容，并按以下JSON格式输出：
```json
{{
  "event_summary": "事件总结",
  "visual_details": "推测的视觉细节",
  "key_moments": ["关键时刻1", "关键时刻2"],
  "people_objects": "推测的人物和物体",
  "location_time": "推测的地点和时间"
}}
```"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response, usage = self.api_manager.call(
            messages,
            temperature=0.5,
            max_tokens=800
        )

        return self._parse_understanding(response)

    def _parse_understanding(self, response: str) -> Dict[str, Any]:
        """解析理解结果"""
        import re

        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)

        if json_match:
            try:
                understanding = json.loads(json_match.group(1))
            except:
                understanding = {
                    "event_summary": response[:200],
                    "visual_details": "",
                    "key_moments": [],
                    "people_objects": "",
                    "location_time": ""
                }
        else:
            understanding = {
                "event_summary": response[:200],
                "visual_details": response,
                "key_moments": [],
                "people_objects": "",
                "location_time": ""
            }

        # 生成完整描述
        full_description = self._generate_full_description(understanding)
        understanding['full_description'] = full_description

        return understanding

    def _generate_full_description(self, understanding: Dict[str, Any]) -> str:
        """生成完整的视频描述"""
        parts = []

        if understanding.get('event_summary'):
            parts.append(f"事件：{understanding['event_summary']}")

        if understanding.get('visual_details'):
            parts.append(f"视觉细节：{understanding['visual_details']}")

        if understanding.get('people_objects'):
            parts.append(f"人物和物体：{understanding['people_objects']}")

        if understanding.get('location_time'):
            parts.append(f"地点和时间：{understanding['location_time']}")

        if understanding.get('key_moments'):
            parts.append(f"关键时刻：{', '.join(understanding['key_moments'])}")

        return "\n\n".join(parts)


# ==========================================
# 评论树生成器
# ==========================================

class CommentTreeGenerator:
    """评论树生成器"""

    def __init__(
        self,
        num_comments: int = 15,
        model: str = 'gpt-4o-mini',
        max_depth: int = 5,
        use_video_understanding: bool = True,
        understanding_model: str = 'gpt-4o-mini',
        cache_dir: str = None
    ):
        """
        初始化评论树生成器

        Args:
            num_comments: 目标评论总数
            model: 使用的LLM模型
            max_depth: 最大评论深度
            use_video_understanding: 是否使用视频理解
            understanding_model: 视频理解使用的模型
            cache_dir: 缓存目录
        """
        self.num_comments = num_comments
        self.model = model
        self.max_depth = max_depth
        self.use_video_understanding = use_video_understanding
        self.understanding_model = understanding_model
        self.cache_dir = cache_dir
        self.logger = get_logger()

        # 视频理解智能体
        if self.use_video_understanding:
            self.video_understanding_agent = VideoUnderstandingAgent(
                model=understanding_model,
                cache_dir=cache_dir
            )
        else:
            self.video_understanding_agent = None

    def generate_tree(
        self,
        video_context: CommentContext,
        stance: str = 'support'
    ) -> CommentTree:
        """生成完整的评论树"""
        self.logger.info(f"开始生成评论树 (stance={stance}, target={self.num_comments}条)")

        # 第0步：视频理解
        if self.use_video_understanding and hasattr(self, 'video_understanding_agent'):
            self.logger.info("正在理解视频内容...")
            video_understanding = self.video_understanding_agent.understand_video(video_context)
            video_context.video_understanding = video_understanding
            self.logger.info(f"视频理解完成: {video_understanding['event_summary'][:100]}...")

        # 创建智能体
        agents = self._create_agents(stance)

        # 创建评论树
        tree = CommentTree()

        # 第1步：生成主评论
        self._generate_main_comments(tree, agents, video_context)

        # 第2步：生成回复
        current_count = tree.total_comments()

        while current_count < self.num_comments:
            # 选择要回复的评论
            target = self._select_target_for_reply(tree, video_context)

            if target is None:
                self.logger.warning("无法选择回复目标，停止生成")
                break

            # 选择合适的智能体
            agent = self._select_agent_for_reply(target, agents, tree)

            # 获取对话历史
            thread_history = tree.get_thread(target.comment_id)

            # 生成回复
            reply = agent.generate_reply(
                video_context,
                target.content,
                thread_history
            )

            # 添加到树中
            tree.add_reply(target.comment_id, reply, agent.agent_role)
            current_count += 1

            self.logger.info(f"生成进度: {current_count}/{self.num_comments}")

        self.logger.info(f"评论树生成完成: {tree.total_comments()}条评论, 最大深度{tree.max_depth()}")
        return tree

    def _create_agents(self, stance: str) -> Dict[str, BaseAgent]:
        """创建所有智能体"""
        agents = {
            'witness': Witness(stance, self.model),
            'expert': Expert(stance, self.model),
            'knowledge_expert': KnowledgeExpert(stance, self.model),
            'skeptic': Skeptic(stance, self.model),
            'passerby': Passerby(stance, self.model)
        }
        return agents

    def _generate_main_comments(
        self,
        tree: CommentTree,
        agents: Dict[str, BaseAgent],
        video_context: CommentContext
    ):
        """生成主评论（根节点）"""
        # 目击者主评论
        witness_comment = agents['witness'].generate_main_comment(video_context)
        tree.add_main_comment(witness_comment, 'witness')

        # 专家主评论
        expert_comment = agents['expert'].generate_main_comment(video_context)
        tree.add_main_comment(expert_comment, 'expert')

        # 外部知识专家主评论
        knowledge_comment = agents['knowledge_expert'].generate_main_comment(video_context)
        tree.add_main_comment(knowledge_comment, 'knowledge_expert')

        # 质疑者主评论
        skeptic_comment = agents['skeptic'].generate_main_comment(video_context)
        tree.add_main_comment(skeptic_comment, 'skeptic')

        # 1-3个路人简短评论
        num_passerby = random.randint(1, 3)
        for i in range(num_passerby):
            passerby_comment = agents['passerby'].generate_main_comment(video_context)
            tree.add_main_comment(passerby_comment, 'passerby')

        self.logger.info(f"生成了 {tree.total_comments()} 条主评论")

    def _select_target_for_reply(
        self,
        tree: CommentTree,
        video_context: CommentContext
    ):
        """选择要回复的评论"""
        candidates = [
            node for node in tree.nodes.values()
            if node.depth < self.max_depth
        ]

        if not candidates:
            return None

        # 加权随机选择
        weights = []
        for node in candidates:
            weight = 1.0

            if node.agent_role == 'skeptic':
                weight *= 2.0

            weight *= (1.0 / (node.depth + 1))

            num_replies = len(node.children)
            weight *= (1.0 / (num_replies + 1))

            weights.append(weight)

        selected = random.choices(candidates, weights=weights, k=1)[0]

        return selected

    def _select_agent_for_reply(
        self,
        target,
        agents: Dict[str, BaseAgent],
        tree: CommentTree
    ) -> BaseAgent:
        """选择合适的智能体进行回复"""
        if target.agent_role == 'skeptic':
            return random.choice([agents['witness'], agents['expert'], agents['passerby']])
        elif target.agent_role == 'expert':
            return agents['passerby']
        elif target.agent_role == 'witness':
            return random.choice([agents['passerby'], agents['knowledge_expert']])
        else:
            return random.choice(list(agents.values()))


# ==========================================
# ConAttacker主控制器
# ==========================================

class ConAttacker:
    """ConAttacker: 多智能体对抗评论生成框架"""

    def __init__(
        self,
        num_comments: int = 15,
        model: str = 'gpt-4o-mini',
        use_video_understanding: bool = True,
        cache_dir: str = None
    ):
        """
        初始化ConAttacker

        Args:
            num_comments: 目标评论数
            model: 使用的模型
            use_video_understanding: 是否使用视频理解
            cache_dir: 缓存目录
        """
        self.num_comments = num_comments
        self.model = model
        self.use_video_understanding = use_video_understanding
        self.cache_dir = cache_dir
        self.logger = get_logger()

    def generate_comments(
        self,
        video_sample,
        stance: str = 'auto'
    ) -> CommentTree:
        """
        生成对抗性评论

        Args:
            video_sample: 视频样本（VideoSample对象或包含video_id, label, title, transcript, video_path的对象）
            stance: 立场 ('auto', 'support', 'doubt')

        Returns:
            评论树
        """
        # 自动选择立场
        if stance == 'auto':
            stance = 'support' if video_sample.label == 'fake' else 'doubt'

        self.logger.info(f"为视频 {video_sample.video_id} 生成对抗性评论 (stance={stance})")

        # 创建评论上下文
        video_context = CommentContext(
            video_id=video_sample.video_id,
            video_description=self._create_video_description(video_sample),
            transcript=video_sample.transcript or "（无转写文本）",
            label=video_sample.label,
            video_path=video_sample.video_path,
            metadata=video_sample.metadata if hasattr(video_sample, 'metadata') else None
        )

        # 生成评论树
        generator = CommentTreeGenerator(
            num_comments=self.num_comments,
            model=self.model,
            use_video_understanding=self.use_video_understanding,
            cache_dir=self.cache_dir
        )

        return generator.generate_tree(video_context, stance)

    def _create_video_description(self, video_sample) -> str:
        """创建视频描述"""
        parts = []

        if hasattr(video_sample, 'title') and video_sample.title:
            parts.append(f"标题：{video_sample.title}")

        if hasattr(video_sample, 'transcript') and video_sample.transcript:
            parts.append(f"音频转写：{video_sample.transcript}")

        if not parts:
            parts.append(f"视频ID: {video_sample.video_id}")

        return "\n".join(parts)

    def save_comments(
        self,
        comment_tree: CommentTree,
        video_sample,
        output_path: str,
        condition: str = 'adversarial'
    ):
        """保存评论到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确定stance
        stance = 'support' if video_sample.label == 'fake' else 'doubt'

        # 构建输出数据
        data = {
            "video_id": video_sample.video_id,
            "video_label": video_sample.label,
            "fake_type": video_sample.fake_type if hasattr(video_sample, 'fake_type') else None,
            "condition": condition,
            "stance": stance,
            "total_comments": comment_tree.total_comments(),
            "max_depth": comment_tree.max_depth(),
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "tree_structure": {
                "nodes": [node.to_dict() for node in comment_tree.get_all_comments()]
            }
        }

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"评论已保存到: {output_path}")


# ==========================================
# 便捷函数
# ==========================================

def get_stance_from_video_label(video_label: str) -> str:
    """根据视频标签自动选择立场"""
    if video_label == 'fake':
        return 'support'
    elif video_label == 'real':
        return 'doubt'
    else:
        raise ValueError(f"Unknown video label: {video_label}")