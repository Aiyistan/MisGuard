#!/usr/bin/env python3
"""
运行MisGuard检测视频虚假信息

用法:
    python run_misguard.py --data data/sample.jsonl --output outputs/predictions.json
"""
import argparse
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CONFIG, SERPER_API_KEY
from misguard import MisGuard
from dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="运行MisGuard检测视频虚假信息")
    parser.add_argument("--data", type=str, help="数据集路径（JSONL格式）")
    parser.add_argument("--frames_dir", type=str, help="视频帧目录")
    parser.add_argument("--comments_dir", type=str, default="outputs/sample_comments", help="评论目录")
    parser.add_argument("--output", type=str, default="outputs/predictions.json", help="输出文件")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="使用的模型")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--search_api_key", type=str, default=None, help="搜索API密钥")

    args = parser.parse_args()

    # 使用配置文件中的默认值
    data_path = args.data or DATA_CONFIG.get("dataset_path", "data/sample.jsonl")
    frames_dir = args.frames_dir or DATA_CONFIG.get("frames_dir", "data/frames")
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 获取搜索API密钥
    search_api_keys = []
    if args.search_api_key:
        search_api_keys.append(args.search_api_key)
    elif SERPER_API_KEY and SERPER_API_KEY != "YOUR_SEARCH_API_KEY":
        search_api_keys.append(SERPER_API_KEY)

    print(f"加载数据集: {data_path}")
    print(f"视频帧目录: {frames_dir}")
    print(f"评论目录: {args.comments_dir}")
    print(f"输出文件: {output_file}")
    print(f"模型: {args.model}")
    print(f"搜索API: {'已配置' if search_api_keys else '未配置'}")

    # 加载数据集
    dataset = load_dataset(
        data_path=data_path,
        frames_dir=frames_dir,
        max_samples=args.max_samples
    )

    print(f"\n加载了 {len(dataset)} 个视频样本")

    # 创建MisGuard
    defender = MisGuard(
        model=args.model,
        search_api_keys=search_api_keys
    )

    # 评估数据集
    print("\n开始检测...")

    results = defender.evaluate(
        dataset=dataset,
        comment_dir=args.comments_dir,
        output_file=str(output_file)
    )

    # 计算准确率
    correct = sum(1 for r in results if r.get("prediction") == r.get("annotation"))
    accuracy = correct / len(results) if results else 0

    print(f"\n完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()