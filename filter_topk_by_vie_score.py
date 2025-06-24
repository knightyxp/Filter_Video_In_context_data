#!/usr/bin/env python3
import json
import os
import argparse

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Filter top-K samples by overall_score and extract from original JSON"
    )
    parser.add_argument(
        "--task_name", "-t",
        type=str,
        default="obj_addition",
        help="任务名称，对应 vie_score_<task_name>.json 和 updated_data_<task_name>_videos.json"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=10000,
        help="要筛选的样本数量"
    )
    parser.add_argument(
        "--base_dir", "-b",
        type=str,
        default="/projects/D2DCRC/xiangpeng/Senorita",
        help="项目根目录"
    )
    args = parser.parse_args()

    # 路径配置
    score_path = os.path.join(args.base_dir, f"vie_score_{args.task_name}.json")
    orig_path  = os.path.join(
        args.base_dir,
        f"updated_data_{args.task_name}.json"
    )
    out_path   = os.path.join(
        args.base_dir,
        f"clean_top{args.top_k}_{args.task_name}.json"
    )

    # 1. 读取打分结果
    score_data = load_json(score_path)
    if isinstance(score_data, list):
        results = score_data
    else:
        results = score_data.get("results", [])
        
    # 2. 按 overall_score 排序并取前 top_k
    top_samples = sorted(
        results,
        key=lambda x: x.get("overall_score", 0.0),
        reverse=True
    )[: args.top_k]  

    # 3. 读取原始数据
    orig = load_json(orig_path)
    if isinstance(orig, list):
        orig_entries = orig
    else:
        orig_entries = orig.get("results", [])

    # 建立映射：(源视频文件名, 目标视频文件名) -> 原始条目
    orig_map = {}
    for e in orig_entries:
        src_bn = os.path.basename(e.get("source_video_path",""))
        tgt_bn = os.path.basename(e.get("target_video_path",""))
        orig_map[(src_bn, tgt_bn)] = e

    # 4. 匹配抽取
    clean_list = []
    for s in top_samples:
        key = (
            os.path.basename(s.get("source_video_path","")),
            os.path.basename(s.get("target_video_path",""))
        )
        entry = orig_map.get(key)
        if entry:
            clean_list.append(entry)
        else:
            # 如果匹配不到，可根据需要打印或记录日志
            print(f"[WARNING] 无匹配项：{key}")

    # 写出干净数据
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean_list, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成前 {len(clean_list)} 条干净样本：{out_path}")

if __name__ == "__main__":
    main()
