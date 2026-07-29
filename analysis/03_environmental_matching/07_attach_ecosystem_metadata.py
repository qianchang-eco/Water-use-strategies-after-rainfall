# ---------------------------------------------------
# match_ecosystem.py — 生成含生态系统类别的 q 结果
# 输出列包含：
# file, event_id, start, end, length, theta_w, theta_star, ETmax, q, r2,
# SITE_ID, depth_from, Sensor_ID, z_m, LC, ecosystem
# ---------------------------------------------------

import os
import pandas as pd
import re

# -------------------------
# 路径设置（⚠ 请确认这两个路径正确）
# -------------------------
q_results_path = "/data/workspace/hywang/changqian/results_q/q_results.csv"
author_meta_path = "/data/workspace/hywang/changqian/data/ismn_results_star_th0_04may.csv"
output_dir = "/data/workspace/hywang/changqian/results_q/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 读取 q_results.csv
# -------------------------
df_q = pd.read_csv(q_results_path)


# =======================================================
# 1) 从 file 解析 SITE_ID
# =======================================================
def extract_site_id(filename):
    base = os.path.basename(filename)

    # 按 "_sm" 截断
    if "_sm" in base:
        left = base.split("_sm")[0]
    else:
        left = base

    parts = [p for p in left.split("_") if p.strip() != ""]

    # 去掉重复 network（如 AMMA-CATCH_AMMA-CATCH）
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = parts[1:]

    # SITE_ID = 最后一个字段
    return parts[-1]


# =======================================================
# 2) 从 file 提取 depth_from
#    格式如 "0.050000_0.050000"
# =======================================================
def extract_depth(filename):
    m = re.search(r"_([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)_", filename)
    if m:
        return float(m.group(1))
    return None


# =======================================================
# 3) 从 file 提取 Sensor_ID
#    即 depth 之后到日期之前的字段
# =======================================================
def extract_sensor(filename):
    m = re.search(r"[0-9]+\.[0-9]+_[0-9]+\.[0-9]+_(.+?)_20[0-9]{6}", filename)
    if m:
        return m.group(1)
    return None


# 添加解析列
df_q["SITE_ID"] = df_q["file"].apply(extract_site_id)
df_q["depth_from"] = df_q["file"].apply(extract_depth)
df_q["Sensor_ID"] = df_q["file"].apply(extract_sensor)


# =======================================================
# 4) 读取作者 ismn_results_star_th0_04may.csv（含 LC）
# =======================================================
df_meta = pd.read_csv(author_meta_path)

# 只取需要的字段（避免重复）
df_meta_sub = df_meta[["SITE_ID", "z_m", "LC"]].drop_duplicates()


# =======================================================
# 5) 进行 merge
# =======================================================
df = df_q.merge(df_meta_sub, on="SITE_ID", how="left")


# =======================================================
# 6) LC → ecosystem 映射规则
# =======================================================
def lc_to_ecosystem(x):
    try:
        x = int(x)
    except:
        return "unknown"

    # forest
    if x in {50, 60, 61, 70, 71, 72, 80, 81, 82}:
        return "forest"

    # grassland
    if x in {130}:
        return "grassland"

    # savanna
    if x in {100, 110}:
        return "savanna"

    return "other"

df["ecosystem"] = df["LC"].apply(lc_to_ecosystem)


# =======================================================
# 7) 保存结果
# =======================================================
out_path = os.path.join(output_dir, "q_results_with_ecosystem.csv")
df.to_csv(out_path, index=False)
print("完成！输出文件：", out_path)

print("\n生态系统分类统计：")
print(df["ecosystem"].value_counts())

