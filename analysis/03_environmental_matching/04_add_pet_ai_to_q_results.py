import os
import pandas as pd
import numpy as np

# =========================
# 路径配置
# =========================
Q_FILE = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc_latlon.csv"
PET_DIR = "/data/workspace/hywang/changqian/data/pet/csv"

OUT_FILE = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc_latlon_pet_ai.csv"

# =========================
# 1) 读取主表
# =========================
df = pd.read_csv(Q_FILE)

required_cols = {"SITE_ID", "start", "end", "precip"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 时间字段
df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])

# 新列初始化
df["pet"] = np.nan
df["ai"] = np.nan

# =========================
# 2) PET 文件缓存（避免重复读取）
# =========================
pet_cache = {}

def load_pet(site_id):
    """按 SITE_ID 读取 PET CSV，并缓存"""
    if site_id in pet_cache:
        return pet_cache[site_id]

    pet_file = os.path.join(PET_DIR, f"{site_id}.csv")
    if not os.path.exists(pet_file):
        pet_cache[site_id] = None
        return None

    pdf = pd.read_csv(pet_file)
    if "date" not in pdf.columns or "pet" not in pdf.columns:
        pet_cache[site_id] = None
        return None

    pdf["date"] = pd.to_datetime(pdf["date"])
    pet_cache[site_id] = pdf
    return pdf

# =========================
# 3) 逐行计算 event-mean PET
# =========================
for idx, row in df.iterrows():
    site_id = str(row["SITE_ID"])
    start = row["start"]
    end = row["end"]
    precip = row["precip"]

    pdf = load_pet(site_id)
    if pdf is None:
        continue

    # 时间窗口
    sub = pdf[(pdf["date"] >= start) & (pdf["date"] <= end)]
    if sub.empty:
        continue

    pet_mean = sub["pet"].mean(skipna=True)
    df.at[idx, "pet"] = pet_mean

    # AI = PET / P
    if pd.notna(precip) and precip > 0:
        df.at[idx, "ai"] = pet_mean / precip
    else:
        df.at[idx, "ai"] = np.nan

# =========================
# 4) 保存结果
# =========================
df.to_csv(OUT_FILE, index=False)

print("Finished.")
print(f"Output saved to:\n{OUT_FILE}")

