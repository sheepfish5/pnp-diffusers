import os
import shutil

# 设置你的 input_dir 路径
input_dir = "input_dir"

# 四个季节的文件夹
seasons = ["spring", "summer", "autumn", "winter"]

for season in seasons:
    season_dir = os.path.join(input_dir, season)
    if not os.path.isdir(season_dir):
        continue

    for filename in os.listdir(season_dir):
        if filename.endswith(".jpg") and len(filename) == 6:  # 确保是格式如 "xx.jpg"
            new_filename = f"{season}-{filename}"
            src_path = os.path.join(season_dir, filename)
            dst_path = os.path.join(input_dir, new_filename)
            shutil.move(src_path, dst_path)

    # 可选：删除空的季节文件夹
    # os.rmdir(season_dir)
