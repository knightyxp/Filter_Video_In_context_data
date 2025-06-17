import os
import subprocess
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from tqdm import tqdm

# 支持的视频格式
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.wmv', '.mpeg', '.mpg')

def get_video_files(root_dir):
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(VIDEO_EXTS):
                video_files.append(os.path.join(dirpath, fname))
    return video_files

def get_resolution(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x', video_path
        ]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return (os.path.dirname(video_path), res) if res else (os.path.dirname(video_path), None)
    except Exception:
        return (os.path.dirname(video_path), None)

def main(root_dir, num_workers=None):
    video_files = get_video_files(root_dir)
    print(f"共找到 {len(video_files)} 个视频文件，开始统计分辨率...")

    # 多进程处理
    with Pool(num_workers or cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(get_resolution, video_files), total=len(video_files)))

    # 按目录统计分辨率分布
    dir_resolution_counter = defaultdict(Counter)
    for dirpath, res in results:
        if res:
            dir_resolution_counter[dirpath][res] += 1
        else:
            dir_resolution_counter[dirpath]['Unknown'] += 1

    # 输出结果
    for dirpath, counter in dir_resolution_counter.items():
        print(f"\n目录: {dirpath}")
        for res, count in counter.most_common():
            print(f"  {res}: {count} 个")

if __name__ == '__main__':
    # 替换为你的 Senorita 目录绝对路径
    dir_path = '/projects/D2DCRC/xiangpeng/Senorita/obj_swap_upload'
    main(dir_path)