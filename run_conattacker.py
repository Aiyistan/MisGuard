#!/usr/bin/env python3
"""
运行ConAttacker生成对抗性评论

用法:
    python run_conattacker.py --data data/sample.jsonl --output outputs/comments/
"""
import argparse
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CONFIG
from conattacker import ConAttacker
from dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="运行ConAttacker生成对抗性评论")
    parser.add_argument("--data", type=str, help="数据集路径（JSONL格式）")
    parser.add_argument("--frames_dir", type=str, help="视频帧目录")
    parser.add_argument("--output", type=str, default="outputs/sample_comments", help="输出目录")
    parser.add_argument("--num_comments", type=int, default=15, help="每个视频生成的评论数")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="使用的模型")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--stance", type=str, default="auto", choices=["auto", "support", "doubt"], help="评论立场")

    args = parser.parse_args()

    # 使用配置文件中的默认值
    data_path = args.data or DATA_CONFIG.get("dataset_path", "data/sample.jsonl")
    frames_dir = args.frames_dir or DATA_CONFIG.get("frames_dir", "data/frames")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据集: {data_path}")
    print(f"视频帧目录: {frames_dir}")
    print(f"输出目录: {output_dir}")
    print(f"评论数: {args.num_comments}")
    print(f"模型: {args.model}")
    print(f"立场: {args.stance}")

    # 加载数据集
    dataset = load_dataset(
        data_path=data_path,
        frames_dir=frames_dir,
        max_samples=args.max_samples
    )

    print(f"\n加载了 {len(dataset)} 个视频样本")

    # 创建ConAttacker
    attacker = ConAttacker(
        num_comments=args.num_comments,
        model=args.model,
        use_video_understanding=True
    )

    # 为每个视频生成评论
    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] 处理视频: {sample.video_id}")

        try:
            # 生成评论
            comment_tree = attacker.generate_comments(sample, stance=args.stance)

            print(f"  生成了 {comment_tree.total_comments()} 条评论")

            # 保存评论
            output_path = output_dir / f"{sample.video_id}_comments.json"
            attacker.save_comments(comment_tree, sample, str(output_path))

        except Exception as e:
            print(f"  错误: {e}")
            continue

    print(f"\n完成! 评论已保存到: {output_dir}")


if __name__ == "__main__":
    main()