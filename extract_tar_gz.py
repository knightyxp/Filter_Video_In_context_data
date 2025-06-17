import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# 需要检查的子文件夹
SUBDIRS = [
    "inpainting_upload",
    "obj_removal_videos_upload",
    "obj_removal_videos_upload2",
    "obj_swap_upload",
    "outpainting"
]

BASE_DIR = "."  # 你的根目录

def is_fully_extracted(tar_path):
    """判断tar.gz里的所有文件是否都已存在于解压目录"""
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            target_path = os.path.join(os.path.dirname(tar_path), member.name)
            if not os.path.exists(target_path):
                return False
    return True

def find_targz_files(root):
    """递归查找所有tar.gz文件"""
    targz_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.tar.gz'):
                full_path = os.path.join(dirpath, fname)
                print(f"Checking: {full_path}")
                if not is_fully_extracted(full_path):
                    targz_files.append(full_path)
    return targz_files

def extract_targz(tar_path):
    try:
        extract_dir = os.path.splitext(os.path.splitext(tar_path)[0])[0]
        print(f"解压: {tar_path} -> {extract_dir}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(tar_path))
        print(f"完成: {tar_path}")
        return tar_path, True
    except Exception as e:
        print(f"解压失败: {tar_path}, 错误: {e}")
        return tar_path, False

def main():
    all_targz = []
    for sub in SUBDIRS:
        sub_path = os.path.join(BASE_DIR, sub)
        print(f"检查目录: {sub_path}")
        if os.path.exists(sub_path):
            files = find_targz_files(sub_path)
            print(f"{sub_path} 找到 {len(files)} 个 tar.gz")
            all_targz.extend(files)
        else:
            print(f"目录不存在: {sub_path}")

    print(f"待解压 tar.gz 文件数: {len(all_targz)}")
    print(all_targz)

    # 多线程解压
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_targz, path) for path in all_targz]
        for future in as_completed(futures):
            tar_path, success = future.result()
            if not success:
                print(f"失败: {tar_path}")

if __name__ == "__main__":
    main()