import pandas as pd
import os

# ========================
# 路径定义
# ========================
q_file = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem.csv"
data_dir = "/data/workspace/hywang/changqian/code/ismn_reader/merged_ismn_chirps"

# ========================
# 读取 q 结果
# ========================
q_df = pd.read_csv(q_file)

# ------------------------
# 1. ecosystem 筛选
# ------------------------
valid_ecosystems = ["forest", "grassland", "savanna"]
q_df = q_df[q_df["ecosystem"].isin(valid_ecosystems)].copy()

# ------------------------
# 2. 时间字段转 datetime
# ------------------------
q_df["start"] = pd.to_datetime(q_df["start"])
q_df["end"] = pd.to_datetime(q_df["end"])

# ------------------------
# 3. 新列初始化
# ------------------------
q_df["precip"] = pd.NA
q_df["SWC"] = pd.NA

# ========================
# 4. 按 file + 时间窗逐行计算
# ========================
for idx, row in q_df.iterrows():
    file_name = row["file"]
    start = row["start"]
    end = row["end"]

    csv_path = os.path.join(data_dir, file_name)

    if not os.path.exists(csv_path):
        print(f"[WARNING] Missing file: {csv_path}")
        continue

    # 读取对应 ISMN + CHIRPS 文件
    df = pd.read_csv(csv_path)

    # 时间字段
    df["DATE"] = pd.to_datetime(df["DATE"])

    # 事件窗口
    mask = (df["DATE"] >= start) & (df["DATE"] <= end)
    sub = df.loc[mask]

    if sub.empty:
        continue

    # 计算
    q_df.at[idx, "precip"] = sub["precip"].sum(skipna=True)
    q_df.at[idx, "SWC"] = sub["SWC"].mean(skipna=True)

# ========================
# 5. 保存结果
# ========================
out_path = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc.csv"
q_df.to_csv(out_path, index=False)

print(f"Finished. Output saved to:\n{out_path}")

