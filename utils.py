"""
工具模块
合并了API管理器和日志功能
"""
import time
import logging
import sys
import json
import re
import base64
import requests
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    MODEL_CONFIG,
    get_model_config
)


# ==========================================
# 日志系统
# ==========================================

class Logger:
    """统一的日志记录器"""

    _instance: Optional['Logger'] = None

    def __new__(cls, name: str = "MisGuard"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name: str = "MisGuard"):
        if self._initialized:
            return

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        self._initialized = True

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


def get_logger(name: str = "MisGuard") -> Logger:
    """获取日志记录器实例"""
    return Logger(name)


# ==========================================
# API管理器
# ==========================================

class APIManager:
    """统一的API调用管理器"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        初始化API管理器

        Args:
            model: 模型名称
        """
        self.model = model
        self.config = get_model_config(model)
        self.logger = get_logger()

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

        # 限流控制
        self.last_call_time = 0
        self.rate_limit_delay = 0.5

    def _wait_for_rate_limit(self):
        """等待以满足限流要求"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 60
    ) -> Tuple[str, Dict[str, int]]:
        """
        调用API

        Args:
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间（秒）

        Returns:
            (response_text, usage_info)
        """
        self._wait_for_rate_limit()

        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 4096)

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )

                text = response.choices[0].message.content or ""
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }

                return text, usage

            except Exception as e:
                self.logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise e

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def create_image_message(self, image_path: str, text: str = "") -> Dict[str, Any]:
        """创建包含图片的消息"""
        base64_image = self.encode_image_to_base64(image_path)

        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

        return {"role": "user", "content": content}


def create_api_manager(model: str = "gpt-4o-mini") -> APIManager:
    """创建API管理器的便捷函数"""
    return APIManager(model)


# ==========================================
# 搜索工具
# ==========================================

class SearchTool:
    """网络搜索工具"""

    def __init__(self, api_keys: List[str] = None):
        """
        初始化搜索工具

        Args:
            api_keys: Serper API密钥列表
        """
        self.api_keys = api_keys or []
        self.logger = get_logger()

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行搜索

        Args:
            query: 搜索查询
            num_results: 结果数量

        Returns:
            搜索结果列表
        """
        if not self.api_keys:
            self.logger.warning("没有配置搜索API密钥")
            return []

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": num_results})

        keys_to_try = self.api_keys.copy()
        random.shuffle(keys_to_try)

        for api_key in keys_to_try:
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("organic", []) + data.get("news", [])
            except Exception as e:
                self.logger.warning(f"搜索请求失败: {e}")
                continue

        return []

    def format_results(self, results: List[Dict], max_results: int = 4) -> str:
        """格式化搜索结果"""
        if not results:
            return "No relevant search results found."

        formatted = []
        for item in results[:max_results]:
            title = item.get("title", "N/A")
            snippet = item.get("snippet", "N/A")
            date = item.get("date", "Unknown")
            source = item.get("source", "Unknown")

            formatted.append(
                f"Source: {source} ({date})\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}"
            )

        return "\n\n".join(formatted)


# ==========================================
# 视频处理工具
# ==========================================

class VideoProcessor:
    """视频预处理器"""

    def __init__(self, num_frames: int = 8):
        """
        初始化视频处理器

        Args:
            num_frames: 要提取的帧数
        """
        self.num_frames = num_frames
        self.logger = get_logger()

    def load_frames_from_directory(
        self,
        frames_dir: str,
        num_frames: Optional[int] = None,
        extensions: list = None
    ) -> List[str]:
        """
        从目录获取已提取的视频帧路径

        Args:
            frames_dir: 帧目录路径
            num_frames: 要获取的帧数
            extensions: 文件扩展名列表

        Returns:
            帧文件路径列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']

        frames_dir = Path(frames_dir)

        if not frames_dir.exists():
            self.logger.error(f"帧目录不存在: {frames_dir}")
            return []

        frame_files = []
        for ext in extensions:
            files = sorted(frames_dir.glob(f"*_frame_*{ext}"))
            if not files:
                files = sorted(frames_dir.glob(f"*{ext}"))
            frame_files.extend(files)

        if not frame_files:
            self.logger.error(f"帧目录中没有找到帧文件: {frames_dir}")
            return []

        # 限制帧数
        num_frames = num_frames or self.num_frames
        if len(frame_files) > num_frames:
            import numpy as np
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        return [str(f) for f in frame_files]

    def frame_to_base64(self, frame_path: str) -> str:
        """将帧文件转换为base64"""
        try:
            with open(frame_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"读取帧文件失败 {frame_path}: {e}")
            return ""

    def compress_image_base64(self, base64_str: str, max_size: int = 512) -> str:
        """压缩base64图片"""
        if Image is None:
            return base64_str

        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"图片压缩失败: {e}")
            return base64_str


# ==========================================
# JSON解析工具
# ==========================================

def extract_json(text: str) -> Dict:
    """从文本中提取JSON"""
    if not text:
        return {}

    # 尝试找到JSON块
    start = text.find("{")
    if start != -1:
        stack = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                stack += 1
            elif text[i] == "}":
                stack -= 1
                if stack == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break

    # 尝试正则匹配
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {}

    return {}


def normalize_verdict(verdict: str) -> str:
    """标准化判决结果"""
    v = str(verdict).strip().upper()
    if "FAKE" in v:
        return "FAKE"
    if "REAL" in v:
        return "REAL"
    return "Unknown"


def normalize_risk(risk: str) -> str:
    """标准化风险等级"""
    v = str(risk).strip().upper()
    if "HIGH" in v:
        return "HIGH"
    if "MEDIUM" in v:
        return "MEDIUM"
    if "LOW" in v:
        return "LOW"
    return "UNKNOWN"