import os
import time
import pandas as pd
import xarray as xr
import numpy as np

# =========================
# 配置区：按你的路径写死
# =========================
MASTER_CSV = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem_precip_swc_latlon.csv"
PET_NC_DIR = "/data/workspace/hywang/changqian/data/pet"
OUT_DIR = "/data/workspace/hywang/changqian/data/pet/csv"

YEARS = list(range(2001, 2025))                 # 2001-2024
NC_PATTERN = "{year}_daily_pet.nc"              # 2001_daily_pet.nc
SEL_METHOD = "nearest"                          # 最近邻

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def pick_pet_var(ds: xr.Dataset) -> str:
    """自动识别PET变量名：优先 pet/PET/Pet，否则取第一个 data_var。"""
    for cand in ["pet", "PET", "Pet"]:
        if cand in ds.data_vars:
            return cand
    return list(ds.data_vars.keys())[0]

def normalize_latlon_coords(da: xr.DataArray) -> xr.DataArray:
    """兼容 latitude/longitude 命名。"""
    coord_map = {}
    if "latitude" in da.coords and "lat" not in da.coords:
        coord_map["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords:
        coord_map["longitude"] = "lon"
    if coord_map:
        da = da.rename(coord_map)
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"Cannot find lat/lon coords. Coords: {list(da.coords)}")
    return da

def read_resume_start_date(out_csv: str):
    """
    若站点 CSV 已存在，则返回已写入的最后日期（datetime.date），用于断点续跑；
    否则返回 None。
    """
    if not os.path.exists(out_csv):
        return None
    try:
        # 只读最后一行：pandas 没有直接 tail 读文件末行的轻量接口，
        # 这里采用读最后几行折中方案（文件不大，且只一次/站点）。
        df_tail = pd.read_csv(out_csv).tail(5)
        if df_tail.empty or "date" not in df_tail.columns:
            return None
        last_date_str = str(df_tail["date"].dropna().iloc[-1])
        last_date = pd.to_datetime(last_date_str).date()
        return last_date
    except Exception:
        return None

def append_site_year(out_csv: str, site_lon: float, site_lat: float, dates, pets, write_header: bool):
    """将某站点某些日期的 pet 序列追加写入 CSV。"""
    out = pd.DataFrame({
        "lon": site_lon,
        "lat": site_lat,
        "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "pet": pets
    })
    out.to_csv(out_csv, mode="a", index=False, header=write_header)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # 1) 读取主表，去重得到站点经纬度
    # -------------------------
    dfm = pd.read_csv(MASTER_CSV)
    need_cols = {"SITE_ID", "lat", "lon"}
    missing = need_cols - set(dfm.columns)
    if missing:
        raise ValueError(f"MASTER_CSV missing columns: {missing}")

    sites = (
        dfm[["SITE_ID", "lat", "lon"]]
        .dropna(subset=["SITE_ID", "lat", "lon"])
        .drop_duplicates(subset=["SITE_ID"])
        .reset_index(drop=True)
    )
    if sites.empty:
        raise ValueError("No valid SITE_ID/lat/lon found in master CSV.")

    # 站点列表
    sites["SITE_ID"] = sites["SITE_ID"].astype(str)
    log(f"Loaded master sites: {len(sites)} sites.")
    log(f"PET NC dir: {PET_NC_DIR}")
    log(f"Output dir: {OUT_DIR}")

    # -------------------------
    # 2) 预扫描：记录每个站点是否已存在与断点日期
    # -------------------------
    resume_info = {}
    for _, r in sites.iterrows():
        sid = r["SITE_ID"]
        out_csv = os.path.join(OUT_DIR, f"{sid}.csv")
        last_date = read_resume_start_date(out_csv)
        resume_info[sid] = {
            "out_csv": out_csv,
            "exists": os.path.exists(out_csv),
            "last_date": last_date  # None 表示从头开始
        }

    # -------------------------
    # 3) 年度循环：每年只打开一次 NetCDF，然后对所有站点逐个点提取
    # -------------------------
    t0 = time.time()

    for yi, year in enumerate(YEARS, start=1):
        nc_path = os.path.join(PET_NC_DIR, NC_PATTERN.format(year=year))
        if not os.path.exists(nc_path):
            log(f"[WARNING] Missing NC file: {nc_path} (skip)")
            continue

        log(f"Year {year} ({yi}/{len(YEARS)}): opening {nc_path}")

        ds = xr.open_dataset(nc_path, decode_times=True)
        pet_var = pick_pet_var(ds)
        da = normalize_latlon_coords(ds[pet_var])

        # 确保 time 坐标存在
        if "time" not in da.coords:
            raise ValueError(f"No 'time' coord in {nc_path}. Coords: {list(da.coords)}")

        # 这一年所有日期（datetime64）
        year_times = pd.to_datetime(da["time"].values)

        # 对每个站点逐个提取
        wrote_sites = 0
        skipped_sites = 0

        for _, r in sites.iterrows():
            sid = r["SITE_ID"]
            site_lat = float(r["lat"])
            site_lon = float(r["lon"])
            out_csv = resume_info[sid]["out_csv"]

            # 断点续跑：如果该站点已经写到 >= 本年最后一天，则跳过本年
            last_date = resume_info[sid]["last_date"]
            if last_date is not None:
                # 如果该站点最后日期已经 >= 本年最后一天，直接跳过
                if last_date >= year_times.max().date():
                    skipped_sites += 1
                    continue

            # 若该站点已有 last_date，则只提取 last_date+1 之后的日期
            if last_date is not None:
                mask = year_times.date > last_date
                if not mask.any():
                    skipped_sites += 1
                    continue
                times_to_get = year_times[mask]
            else:
                times_to_get = year_times

            # 关键：单点 sel，避免 vectorized 大索引灾难
            # 先选 lat/lon，再按 time 子集
            try:
                point = da.sel(lat=site_lat, lon=site_lon, method=SEL_METHOD)
                point = point.sel(time=times_to_get.values)
                pets = point.values
            except Exception as e:
                log(f"[WARNING] SITE {sid} year {year} failed: {repr(e)}")
                continue

            # 写入：如果文件不存在则写 header，否则追加不写
            write_header = not os.path.exists(out_csv)
            append_site_year(out_csv, site_lon, site_lat, times_to_get.values, pets, write_header)

            # 更新断点日期
            resume_info[sid]["last_date"] = pd.to_datetime(times_to_get.values[-1]).date()
            wrote_sites += 1

            # 轻量进度输出（每 50 个站点打印一次，避免日志过多）
            if wrote_sites % 50 == 0:
                elapsed = time.time() - t0
                log(f"Year {year}: processed {wrote_sites}/{len(sites)} sites (skipped {skipped_sites}); elapsed {elapsed/60:.1f} min")

        ds.close()

        elapsed = time.time() - t0
        log(f"Year {year} DONE: var='{pet_var}', wrote {wrote_sites} sites, skipped {skipped_sites}. Elapsed {elapsed/60:.1f} min")

    log("All done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {repr(e)}")
        raise

