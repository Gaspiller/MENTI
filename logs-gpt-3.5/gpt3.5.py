import os
import re

# 修改为你的日志文件所在文件夹路径
logs_dir = "E:\MENTI\logs-gpt-3.5"

for filename in os.listdir(logs_dir):
    if filename.endswith(".log") and filename.startswith("menti_"):
        # 匹配形如 menti_3.5_2.log
        m = re.match(r"menti_(\d+\.\d+)_(\d+)\.log$", filename)
        if m:
            middle_num = m.group(1)  # 3.5
            last_num = int(m.group(2))  # 2
            new_last_num = last_num - 1
            new_name = f"{middle_num}_{new_last_num}.log"
            old_path = os.path.join(logs_dir, filename)
            new_path = os.path.join(logs_dir, new_name)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")

