import os
import pandas as pd
from detect_drydown_events import detect_drydown_events

# 输入目录（请按你的路径修改）
input_dir = "/data/workspace/hywang/changqian/code/ismn_reader/merged_ismn_chirps/"

# 输出目录
output_dir = "/data/workspace/hywang/changqian/results_drydowns/"
os.makedirs(output_dir, exist_ok=True)

all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

print(f"共检测到 CSV 文件：{len(all_files)}")

for idx, file in enumerate(all_files, 1):
    print(f"[{idx}/{len(all_files)}] 处理 {file}")

    file_path = os.path.join(input_dir, file)

    try:
        df = pd.read_csv(file_path)

        # 字段检查
        if not {"DATE", "SWC"}.issubset(df.columns):
            print(f"跳过：{file} 缺少 DATE 或 SWC")
            continue

        events = detect_drydown_events(df)

        # 将每个事件输出到一个表格
        out_rows = []
        for i, e in enumerate(events):
            out_rows.append({
                "file": file,
                "event_id": i+1,
                "start": e["start"],
                "end": e["end"],
                "length": e["length"]
            })

        if len(out_rows) > 0:
            out_df = pd.DataFrame(out_rows)
        else:
            out_df = pd.DataFrame(columns=["file", "event_id", "start", "end", "length"])

        # 输出文件名
        out_file = os.path.join(output_dir, file.replace(".csv", "_drydowns.csv"))
        out_df.to_csv(out_file, index=False)

        print(f"→ 完成事件识别，共 {len(out_rows)} 个事件，已保存：{out_file}")

    except Exception as e:
        print(f"处理文件 {file} 时出错：{e}")
