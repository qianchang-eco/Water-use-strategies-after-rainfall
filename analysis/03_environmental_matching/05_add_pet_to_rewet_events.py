import os
import pandas as pd
import numpy as np

# =========================
# 路径配置
# =========================
REWET_FILE = "/data/workspace/hywang/changqian/results_q/rewet_q_results_time.csv"
Q_FILE = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc_latlon_pet_ai.csv"
PET_DIR = "/data/workspace/hywang/changqian/data/pet/csv"

OUT_FILE = "/data/workspace/hywang/changqian/results_q/rewet_q_results_time_pet.csv"

# =========================
# 1) 读取数据
# =========================
rewet = pd.read_csv(REWET_FILE)
qdf = pd.read_csv(Q_FILE)

# -------------------------
# ecosystem 筛选
# -------------------------
valid_ecos = ["forest", "grassland", "savanna"]
rewet = rewet[rewet["ecosystem"].isin(valid_ecos)].copy()

# -------------------------
# file -> SITE_ID, pet 映射
# -------------------------
map_cols = qdf[["file", "SITE_ID", "pet"]].drop_duplicates()
rewet = rewet.merge(map_cols, on="file", how="left")

# =========================
# 时间字段处理
# =========================
for col in rewet.columns:
    if col.endswith("_start") or col.endswith("_end"):
        rewet[col] = pd.to_datetime(rewet[col], errors="coerce")

# =========================
# PET 文件缓存
# =========================
pet_cache = {}

def load_pet(site_id):
    if site_id in pet_cache:
        return pet_cache[site_id]

    pet_file = os.path.join(PET_DIR, f"{site_id}.csv")
    if not os.path.exists(pet_file):
        pet_cache[site_id] = None
        return None

    df = pd.read_csv(pet_file)
    df["date"] = pd.to_datetime(df["date"])
    pet_cache[site_id] = df
    return df

# =========================
# 计算 pet_X 和 Δpet_X
# =========================
for X in [7, 14, 21]:
    rewet[f"pet_{X}"] = np.nan
    rewet[f"dpet_{X}"] = np.nan

    for idx, row in rewet.iterrows():
        site_id = row["SITE_ID"]
        pet_event = row["pet"]

        start = row.get(f"q_{X}_start")
        end = row.get(f"q_{X}_end")

        if pd.isna(site_id) or pd.isna(start) or pd.isna(end):
            continue

        pdf = load_pet(site_id)
        if pdf is None:
            continue

        sub = pdf[(pdf["date"] >= start) & (pdf["date"] <= end)]
        if sub.empty:
            continue

        pet_X = sub["pet"].mean(skipna=True)
        rewet.at[idx, f"pet_{X}"] = pet_X

        if pd.notna(pet_event):
            rewet.at[idx, f"dpet_{X}"] = pet_event - pet_X

# =========================
# 保存结果
# =========================
rewet.to_csv(OUT_FILE, index=False)

print("Finished.")
print(f"Output saved to:\n{OUT_FILE}")

