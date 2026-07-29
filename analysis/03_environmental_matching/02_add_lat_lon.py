import pandas as pd
import os

# ========================
# 路径定义
# ========================
q_file = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc.csv"
coord_dir = "/data/workspace/hywang/changqian/data/ISMN/ancillary/CHIRPS/extracted"

# ========================
# 读取主表
# ========================
df = pd.read_csv(q_file)

# 新列初始化
df["lat"] = pd.NA
df["lon"] = pd.NA

# ========================
# 按 SITE_ID 提取经纬度
# ========================
for site_id in df["SITE_ID"].unique():
    coord_file = os.path.join(coord_dir, f"{site_id}.csv")

    if not os.path.exists(coord_file):
        print(f"[WARNING] Missing coordinate file: {coord_file}")
        continue

    coord_df = pd.read_csv(coord_file)

    if coord_df.empty:
        continue

    # 取任意一行即可（这里取第二行，若不存在则取第一行）
    row = coord_df.iloc[1] if len(coord_df) > 1 else coord_df.iloc[0]

    lat = row["latitude"]
    lon = row["longitude"]

    # 回填到所有对应 SITE_ID 的行
    df.loc[df["SITE_ID"] == site_id, "lat"] = lat
    df.loc[df["SITE_ID"] == site_id, "lon"] = lon

# ========================
# 保存结果
# ========================
out_path = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc_latlon.csv"
df.to_csv(out_path, index=False)

print("Finished adding lat/lon.")
print(f"Output saved to:\n{out_path}")

