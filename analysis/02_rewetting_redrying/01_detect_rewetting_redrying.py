import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

# ------------------------- #
#       q-model 函数
# ------------------------- #
def q_model(t, theta0, k, q, theta_star, theta_w):
    delta_theta = theta0 - theta_w
    b = delta_theta ** (1 - q)
    a = (1 - q) / ((theta_star - theta_w) ** q)
    return (-k * a * t + b) ** (1/(1-q)) + theta_w


# ------------------------- #
#   寻找返湿 → 新干燥段
# ------------------------- #
def find_new_drydown(df, end_date, window_days=7, p_thresh=0.001, swc_col="SWC"):
    start = end_date + pd.Timedelta(days=1)
    end = end_date + pd.Timedelta(days=window_days)

    sub = df.loc[start:end]
    if len(sub) == 0:
        return None

    # 要有降水
    if (sub["precip"] >= p_thresh).sum() == 0:
        return None

    # 找峰值
    theta_end = df.loc[end_date, swc_col]
    peak_t = sub[swc_col].idxmax()
    theta_peak = sub.loc[peak_t, swc_col]

    if theta_peak <= theta_end:
        return None

    # 峰值后找新干燥段
    search_end = peak_t + pd.Timedelta(days=30)
    seg = df.loc[peak_t:search_end].copy()

    # 大降雨截断
    big_rain = seg[seg["precip"] >= p_thresh]
    if len(big_rain) > 0:
        cut = big_rain.index[0] - pd.Timedelta(days=1)
        seg = seg.loc[:cut]

    # 单调下降检查
    swc = seg[swc_col].values
    diffs = np.diff(swc)
    rebound = np.where(diffs > 0.05)[0]
    if len(rebound) > 0:
        cut = rebound[0] + 1
        seg = seg.iloc[:cut]

    if len(seg) < 3:
        return None

    theta_peak = theta_peak          
    theta_end_dd = seg[swc_col].iloc[-1]
    theta_jump = theta_peak - theta_end_dd

    return {
        "seg": seg,
        "start": seg.index[0],
        "end": seg.index[-1],
        "theta_peak": theta_peak,
        "theta_end": theta_end_dd,
        "theta_jump": theta_jump
    }


# ------------------------- #
#         拟合 q
# ------------------------- #
def fit_q(seg, theta_w, theta_star, z_m, swc_col="SWC"):

    y = seg[swc_col].values
    t = np.arange(len(y))

    theta0 = y[0]

    pet_max = 5      
    k_max = pet_max / (z_m * 1000) * 5

    p0 = [theta0, k_max * 0.2, 1.0]
    bounds = ([theta_w, 0, 0], [theta_star, k_max, 5])

    try:
        popt, _ = curve_fit(
            lambda t, theta0, k, q: q_model(
                t, theta0, k, q, theta_star, theta_w
            ),
            t, y, p0=p0, bounds=bounds, maxfev=10000
        )
        y_pred = q_model(t, *popt, theta_star, theta_w)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        return {
            "q": popt[2],
            "k": popt[1],
            "theta0": popt[0],
            "r2": r2
        }
    except:
        return None


# ------------------------- #
#      主函数（批量）
# ------------------------- #
def compute_rewet_q_all():

    # 已有干旱事件及 q
    qfile = "/data/workspace/hywang/changqian/results_q/q_results_with_ecosystem.csv"
    qdf = pd.read_csv(qfile)

    # 输出
    out_list = []

    for file, subdf in qdf.groupby("file"):
        sensor_file = os.path.join(
            "/data/workspace/hywang/changqian/code/ismn_reader/merged_ismn_chirps",
            file
        )
        if not os.path.exists(sensor_file):
            print("missing:", sensor_file)
            continue

        df = pd.read_csv(sensor_file)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE")

        for idx, row in subdf.iterrows():

            end = pd.to_datetime(row["end"])
            theta_w = row["theta_w"]
            theta_star = row["theta_star"]
            z_m = row["z_m"]

            result = {
                "file": file,
                "event_id": row["event_id"],
                "ecosystem": row["ecosystem"],
                "q_original": row["q"]
            }

            # 7-day / 14-day / 21-day
            for win, label in [(7, "q_7"), (14, "q_14"), (21, "q_21")]:

                out = find_new_drydown(df, end_date=end, window_days=win)

                if out is not None:
                    seg = out["seg"]
                    start_t = out["start"]
                    end_t = out["end"]
                    theta_peak = out["theta_peak"]
                    theta_end_dd = out["theta_end"]
                    theta_jump = out["theta_jump"]

                    res = fit_q(seg, theta_w, theta_star, z_m)

                    if res is not None and res["r2"] >= 0.3:
                        result[label] = res["q"]
                        result[f"{label}_start"] = start_t
                        result[f"{label}_end"] = end_t
                        result[f"{label}_theta_peak"] = theta_peak
                        result[f"{label}_theta_end"] = theta_end_dd
                        result[f"{label}_theta_jump"] = theta_jump
                    else:
                        result[label] = np.nan
                        result[f"{label}_start"] = pd.NaT
                        result[f"{label}_end"] = pd.NaT
                        result[f"{label}_theta_peak"] = np.nan
                        result[f"{label}_theta_end"] = np.nan
                        result[f"{label}_theta_jump"] = np.nan
                else:
                    result[label] = np.nan

            out_list.append(result)

    out = pd.DataFrame(out_list)
    out.to_csv("/data/workspace/hywang/changqian/results_q/rewet_q_results_time.csv", index=False)
    print("Done. Saved to rewet_q_results_time.csv")


if __name__ == "__main__":
    compute_rewet_q_all()

