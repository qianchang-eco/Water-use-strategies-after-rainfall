import os
import glob
import pandas as pd
import xarray as xr
import numpy as np

# ===============================
# 路径设置
# ===============================
q_file = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem.csv"
pet_dir = "/data/workspace/hywang/changqian/data/pet"
chirps_dir = "/data/workspace/hywang/changqian/data/ISMN/ancillary/CHIRPS/extracted"
pet_var = "pet"
out_file = "/data/workspace/hywang/changqian/results_q/rewet_q_results.csv"

# ===============================
# 读取 q 结果
# ===============================
df_q = pd.read_csv(q_file)

# 确保时间格式正确
df_q["start"] = pd.to_datetime(df_q["start"])
df_q["end"] = pd.to_datetime(df_q["end"])

# ===============================
# 打开 PET 数据
# ===============================
pet_files = sorted(
    glob.glob(os.path.join(pet_dir, "*_daily_pet.nc"))
)

ds_pet = xr.open_mfdataset(
    pet_files,
    combine="by_coords",
    parallel=True
)

print(ds_pet)
# ===============================
# 结果容器
# ===============================
pet_list = []
precip_list = []
lat_list = []
lon_list = []

# ===============================
# 主循环（逐个 q 事件）
# ===============================
for idx, row in df_q.iterrows():
    site = row["SITE_ID"]
    t_start = row["start"]
    t_end = row["end"]

    # ---------- 读取 CHIRPS 站点数据 ----------
    chirps_file = os.path.join(chirps_dir, f"{site}.csv")
    if not os.path.exists(chirps_file):
        pet_list.append(np.nan)
        precip_list.append(np.nan)
        lat_list.append(np.nan)
        lon_list.append(np.nan)
        continue

    df_p = pd.read_csv(chirps_file)
    df_p["time"] = pd.to_datetime(df_p["time"])

    # 经纬度（默认整个文件一致）
    lat = df_p["latitude"].iloc[0]
    lon = df_p["longitude"].iloc[0]

    lat_list.append(lat)
    lon_list.append(lon)

    # ---------- 累计降水 ----------
    mask = (df_p["time"] >= t_start) & (df_p["time"] <= t_end)
    precip_sum = df_p.loc[mask, "precip"].sum()
    precip_list.append(precip_sum)

    # ---------- PET（事件期平均） ----------
    pet_event = (
        ds_pet[pet_var]
        .sel(time=slice(t_start, t_end))
        .sel(latitude=lat, longitude=lon, method="nearest")
        .mean()
        .values
    )

    pet_list.append(float(pet_event))

# ===============================
# 拼接结果
# ===============================
df_q["latitude"] = lat_list
df_q["longitude"] = lon_list
df_q["PET"] = pet_list
df_q["precip"] = precip_list

# 明确再保留一次这些列（顺序靠后）
df_q["start"] = df_q["start"]
df_q["end"] = df_q["end"]
df_q["SITE_ID"] = df_q["SITE_ID"]

# ===============================
# 保存
# ===============================
df_q.to_csv(out_file, index=False)

print("Finished!")
print(f"Output saved to: {out_file}")

