#!/usr/bin/env python3
import json
import os
import shutil
import random
import argparse

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="从 JSON 条目中随机抽取样本并拷贝对应视频（自动替换 <task_name>_videos -> <task_name>_videos_upload）"
    )
    parser.add_argument(
        "--json_list", "-j",
        required=True,
        help="输入的 JSON 文件路径（顶层为 list，每项含 source_video_path 和 target_video_path）"
    )
    parser.add_argument(
        "--task_name", "-t",
        required=True,
        help="任务名称，用于替换路径中的 <task_name>_videos 目录"
    )
    parser.add_argument(
        "--video_base_dir", "-v",
        type=str,
        default="",
        help="源码 JSON 里 path 字段对应的文件所在的基目录"
    )
    parser.add_argument(
        "--out_dir", "-o",
        required=True,
        help="拷贝后视频存放的根目录"
    )
    parser.add_argument(
        "--sample_size", "-n",
        type=int,
        default=100,
        help="要随机抽取的样本数量（默认 100）"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="随机种子（可选，用于复现）"
    )
    args = parser.parse_args()

    # 构造目录字符串
    src_dir_name = f"{args.task_name}_videos"
    upload_dir_name = f"{args.task_name}_videos_upload"

    # 创建输出目录和子目录
    src_out = args.out_dir 
    tgt_out = args.out_dir 
    os.makedirs(src_out, exist_ok=True)
    os.makedirs(tgt_out, exist_ok=True)

    # 读取并抽样
    data = load_json(args.json_list)
    if not isinstance(data, list):
        raise ValueError("输入的 JSON 文件顶层应为一个 list")
    if args.sample_size > len(data):
        raise ValueError(f"样本数量 {args.sample_size} 超过可选总数 {len(data)}")
    if args.seed is not None:
        random.seed(args.seed)
    sampled = random.sample(data, args.sample_size)

    # 拷贝文件
    for idx, item in enumerate(sampled, 1):
        # 原始相对路径
        src_rel = item.get("source_video_path", "").lstrip("./")
        tgt_rel = item.get("target_video_path", "").lstrip("./")
        if not src_rel or not tgt_rel:
            print(f"[WARN] 条目缺少路径字段，跳过：{item}")
            continue


        src_rel = src_rel.replace(src_dir_name, upload_dir_name)
        tgt_rel = tgt_rel.replace(src_dir_name, upload_dir_name)

        # 拼接实际路径
        src_path = os.path.join(args.video_base_dir, src_rel)
        tgt_path = os.path.join(args.video_base_dir, tgt_rel)

        # 构造拷贝目标文件名
        src_bn = os.path.basename(src_rel)
        tgt_bn = os.path.basename(tgt_rel)
        dst_src = os.path.join(src_out, f"{idx:03d}_{src_bn}")
        dst_tgt = os.path.join(tgt_out, f"{idx:03d}_{tgt_bn}")

        # 执行拷贝
        try:
            shutil.copy2(src_path, dst_src)
            shutil.copy2(tgt_path, dst_tgt)
        except FileNotFoundError as e:
            print(f"[ERROR] 文件不存在，跳过：{e}")

    print(f"✅ 共抽取并拷贝 {len(sampled)} 条记录。")
    print(f"  源视频放在：{src_out}")
    print(f"  目标视频放在：{tgt_out}")

if __name__ == "__main__":
    main()
