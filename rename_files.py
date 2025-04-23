import os
import argparse


def batch_rename_files(folder_path, name_format, start_number):
    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # 按照默认顺序排列文件

    current_number = start_number

    for file_name in files:
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{name_format}_{current_number:04d}{os.path.splitext(file_name)[1]}"
        new_path = os.path.join(folder_path, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"✅ 重命名: {old_path} -> {new_path}")
            current_number += 1
        except Exception as e:
            print(f"❌ 无法重命名 {old_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量重命名文件夹内的文件")
    parser.add_argument("folder_path", type=str, help="文件夹路径")
    parser.add_argument("name_format", type=str, help="重命名格式（如 spectrum）")
    parser.add_argument("start_number", type=int, help="起始编号（如 41）")

    args = parser.parse_args()

    batch_rename_files(args.folder_path, args.name_format, args.start_number)





