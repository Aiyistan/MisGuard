"""
数据模块
合并了数据加载和处理功能
"""
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass

from config import DATA_CONFIG
from utils import get_logger


# ==========================================
# 类型映射
# ==========================================

TYPE_MAP = {
    # 真实和辟谣
    "真实": "Authentic",
    "辟谣": "Debunking",
    # 虚假类型
    "文本篡改": "Text Tampering",
    "视频篡改": "Video Tampering",
    "音频篡改": "Audio Tampering",
    "逻辑谬误": "Faulty Logic",
    "夸张叙述": "Exaggerated Narration",
    "冒犯性内容": "Offensive Content",
    "AI生成": "AI Generation",
    "知识错误": "Knowledge Error",
    "事件编造": "Event Fabrication",
    "事件拼接": "Event Splicing",
    # 英文保持不变
    "Authentic": "Authentic",
    "Debunking": "Debunking",
    "Text Tampering": "Text Tampering",
    "Video Tampering": "Video Tampering",
    "Audio Tampering": "Audio Tampering",
    "Faulty Logic": "Faulty Logic",
    "Exaggerated Narration": "Exaggerated Narration",
    "Offensive Content": "Offensive Content",
    "AI Generation": "AI Generation",
    "Knowledge Error": "Knowledge Error",
    "Event Fabrication": "Event Fabrication",
    "Event Splicing": "Event Splicing",
}


# ==========================================
# 数据类
# ==========================================

@dataclass
class VideoSample:
    """视频样本数据类"""
    video_id: str
    video_path: str  # 视频帧目录路径
    label: str  # 'real' or 'fake'
    fake_type: Optional[str] = None
    transcript: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "label": self.label,
            "fake_type": self.fake_type,
            "transcript": self.transcript,
            "title": self.title,
            "metadata": self.metadata
        }

    def __post_init__(self):
        """数据清理和标准化"""
        # 标准化标签
        if self.label in ['真', 'real', 'Real', 'REAL']:
            self.label = 'real'
        elif self.label in ['假', 'fake', 'Fake', 'FAKE']:
            self.label = 'fake'

        # 标准化fake_type
        if self.fake_type:
            self.fake_type = TYPE_MAP.get(self.fake_type, self.fake_type)
        else:
            self.fake_type = 'Authentic' if self.label == 'real' else 'Unknown'


# ==========================================
# 数据集类
# ==========================================

class VideoMisinfoDataset:
    """虚假视频检测数据集"""

    def __init__(
        self,
        data_path: str,
        frames_dir: Optional[str] = None,
        filter_label: Optional[str] = None,
        filter_fake_type: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        """
        初始化数据集

        Args:
            data_path: 数据集路径（JSONL格式）
            frames_dir: 视频帧目录
            filter_label: 过滤标签 ('real' or 'fake')
            filter_fake_type: 过滤虚假类型
            max_samples: 最大样本数
            shuffle: 是否打乱数据
            seed: 随机种子
        """
        self.data_path = Path(data_path)
        self.frames_dir = Path(frames_dir) if frames_dir else Path(DATA_CONFIG.get("frames_dir", "data/frames"))
        self.filter_label = filter_label
        self.filter_fake_type = filter_fake_type
        self.max_samples = max_samples
        self.logger = get_logger()

        # 加载数据
        self.samples = self._load_data()

        # 打乱数据
        if shuffle:
            random.seed(seed)
            random.shuffle(self.samples)

        # 限制样本数
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        self.logger.info(f"加载了 {len(self.samples)} 个视频样本")

    def _load_data(self) -> List[VideoSample]:
        """加载数据"""
        samples = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.data_path}")

        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        sample = self._parse_item(data, line_num)

                        if sample and self._should_include(sample):
                            samples.append(sample)

                    except Exception as e:
                        self.logger.error(f"行 {line_num}: 解析错误 - {e}")
                        continue

        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                try:
                    sample = self._parse_item(item, idx)
                    if sample and self._should_include(sample):
                        samples.append(sample)
                except Exception as e:
                    self.logger.error(f"项目 {idx}: 解析错误 - {e}")
                    continue
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")

        return samples

    def _parse_item(self, item: Dict[str, Any], idx: int) -> Optional[VideoSample]:
        """解析数据项"""
        video_id = item.get('video_id') or item.get('id')
        if not video_id:
            self.logger.warning(f"项目 {idx}: 缺少video_id")
            return None

        # 解析标签
        annotation = item.get('annotation', item.get('label', ''))
        if annotation in ['真', 'real', 'Real', 'REAL']:
            label = 'real'
        elif annotation in ['假', 'fake', 'Fake', 'FAKE']:
            label = 'fake'
        else:
            self.logger.warning(f"项目 {idx}: 未知标签 '{annotation}'")
            return None

        # 解析fake_type
        fake_type = item.get('fake_type', item.get('type', label))

        # 帧目录路径
        video_path = str(self.frames_dir / video_id)

        return VideoSample(
            video_id=video_id,
            video_path=video_path,
            label=label,
            fake_type=fake_type,
            transcript=item.get('transcript', ''),
            title=item.get('title', ''),
            metadata=item
        )

    def _should_include(self, sample: VideoSample) -> bool:
        """判断是否应该包含该样本"""
        if self.filter_label and sample.label != self.filter_label:
            return False
        if self.filter_fake_type and sample.fake_type != self.filter_fake_type:
            return False
        return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[VideoSample]:
        return iter(self.samples)

    def get_video_ids(self) -> List[str]:
        """获取所有视频ID"""
        return [sample.video_id for sample in self.samples]

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            "total": len(self.samples),
            "real": sum(1 for s in self.samples if s.label == 'real'),
            "fake": sum(1 for s in self.samples if s.label == 'fake'),
            "with_transcript": sum(1 for s in self.samples if s.transcript),
        }

        fake_type_counts = {}
        for sample in self.samples:
            fake_type = sample.fake_type or 'unknown'
            fake_type_counts[fake_type] = fake_type_counts.get(fake_type, 0) + 1

        stats["by_fake_type"] = fake_type_counts

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print("\n数据集统计信息:")
        print(f"  总样本数: {stats['total']}")
        print(f"  真实视频: {stats['real']} ({stats['real']/stats['total']*100:.1f}%)")
        print(f"  虚假视频: {stats['fake']} ({stats['fake']/stats['total']*100:.1f}%)")
        print(f"  有转写: {stats['with_transcript']}")

        if stats['by_fake_type']:
            print("\n按类型分组:")
            for fake_type, count in sorted(stats['by_fake_type'].items()):
                print(f"  {fake_type}: {count}")


# ==========================================
# 便捷函数
# ==========================================

def load_dataset(
    data_path: str = None,
    frames_dir: str = None,
    **kwargs
) -> VideoMisinfoDataset:
    """
    加载数据集的便捷函数

    Args:
        data_path: 数据集路径
        frames_dir: 帧目录
        **kwargs: 其他参数

    Returns:
        VideoMisinfoDataset对象
    """
    if data_path is None:
        data_path = DATA_CONFIG.get("dataset_path", "data/sample.jsonl")

    if frames_dir is None:
        frames_dir = DATA_CONFIG.get("frames_dir", "data/frames")

    return VideoMisinfoDataset(data_path, frames_dir, **kwargs)


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_video_comments(video_id: str, comment_dir: str = "outputs/sample_comments") -> str:
    """加载视频评论"""
    path = Path(comment_dir) / f"{video_id}_comments.json"

    if not path.exists():
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("tree_structure", {}).get("nodes", [])
        children = {}

        for n in nodes:
            parent = n.get("parent_id")
            if parent not in children:
                children[parent] = []
            children[parent].append(n)

        comments = []

        def dfs(node: Dict, depth: int):
            if len(comments) >= 20:
                return

            role = node.get("agent_role", "")
            text = node.get("content", "")
            indent = "  " * depth
            comments.append(f"{indent}[{role}] {text}")

            for child in children.get(node.get("comment_id"), []):
                dfs(child, depth + 1)

        roots = children.get(None, [])
        for r in roots:
            dfs(r, 0)
            if len(comments) >= 20:
                break

        return "\n".join(comments)

    except Exception as e:
        print(f"Comment load error: {video_id}, {e}")
        return ""


# ==========================================
# 主函数（测试）
# ==========================================

if __name__ == '__main__':
    # 测试数据集加载
    print("测试数据集加载...")

    dataset = load_dataset(
        data_path='data/sample_data/HMVD_with_transcripts.jsonl',
        max_samples=100,
        shuffle=True
    )

    dataset.print_statistics()

    print("\n样本示例:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n{i+1}. {sample.video_id}:")
        print(f"   标签: {sample.label}")
        print(f"   类型: {sample.fake_type}")
        print(f"   标题: {sample.title[:60] if sample.title else '无'}...")