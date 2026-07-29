import os
import pandas as pd
import numpy as np
from fit_q_event import fit_one_event

# 路径（请按你的环境确认）
merged_dir = "/data/workspace/hywang/changqian/code/ismn_reader/merged_ismn_chirps/"
drydown_dir = "/data/workspace/hywang/changqian/results_drydowns/"
output_file = "/data/workspace/hywang/changqian/results_q/q_results.csv"

os.makedirs("/data/workspace/hywang/changqian/results_q/", exist_ok=True)

all_dd_files = [f for f in os.listdir(drydown_dir) if f.endswith("_drydowns.csv")]

results = []

print(f"共检测到 {len(all_dd_files)} 个 drydown 文件。")

for idx, dd_file in enumerate(all_dd_files, 1):
    print(f"[{idx}/{len(all_dd_files)}] 处理 {dd_file}")

    dd_path = os.path.join(drydown_dir, dd_file)
    df_dd = pd.read_csv(dd_path)

    if df_dd.empty:
        continue

    # 找原始文件
    orig_file = df_dd["file"].iloc[0]
    orig_path = os.path.join(merged_dir, orig_file)
    if not os.path.exists(orig_path):
        print(f"原始文件不存在：{orig_file}")
        continue

    df_raw = pd.read_csv(orig_path)
    df_raw["DATE"] = pd.to_datetime(df_raw["DATE"])
    df_raw = df_raw.sort_values("DATE")

    theta_all = df_raw["SWC"].values
    theta_w_site = np.min(theta_all)  # 萎蔫点定义：站点全时段最小值
    theta_fc_site = np.percentile(theta_all, 95)  # 粗略估计上限（可换更精确方法）

    for _, row in df_dd.iterrows():
        start = pd.to_datetime(row["start"])
        end = pd.to_datetime(row["end"])

        event_df = df_raw[(df_raw["DATE"]>=start) & (df_raw["DATE"]<=end)]
        if len(event_df) < 6:
            continue

        theta = event_df["SWC"].values
        t = np.arange(len(theta))

        fit = fit_one_event(
            t=t,
            theta=theta,
            theta_w=theta_w_site,
            theta_fc=theta_fc_site,
            delta_z=0.2
        )

        if fit is None:
            continue

        # 质量控制
        if fit["r2"] < 0.7:
            continue
        if fit["q"] < 0.01:
            continue
        if (fit["theta_star"] - theta_w_site) < 0.005:
            continue

        results.append({
            "file": orig_file,
            "event_id": row["event_id"],
            "start": start,
            "end": end,
            "length": row["length"],
            "theta_w": theta_w_site,
            "theta_star": fit["theta_star"],
            "ETmax": fit["ETmax"],
            "q": fit["q"],
            "r2": fit["r2"]
        })

# 保存
if len(results) > 0:
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"拟合完成！共 {len(results)} 条事件成功拟合 q。")
    print(f"保存到：{output_file}")
else:
    print("没有成功的拟合结果。")

