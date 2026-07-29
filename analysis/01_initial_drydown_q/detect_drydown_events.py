import pandas as pd
import numpy as np
import os

def detect_drydown_events(df, delta_z=0.2, et_threshold=0.5, min_len=6):
    """
    输入：
        df: 具有 ['DATE','SWC'] 两列的数据框
    输出：
        events: 一个事件列表，每个事件包含 start, end, length, df
    """
    df = df.copy()

    # 时间格式化
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")

    # 计算 dθ/dt
    df["d_theta"] = df["SWC"].diff()

    # 计算近似 ET (mm/day): -Δz * dθ/dt * 1000
    df["ET_rate"] = -delta_z * df["d_theta"] * 1000

    events = []
    in_event = False
    start_idx = None

    for i in range(1, len(df)):
        et = df["ET_rate"].iloc[i]
        dtheta = df["d_theta"].iloc[i]

        # ----------------------
        # 事件尚未开始
        # ----------------------
        if not in_event:
            if et >= et_threshold:
                if start_idx is None:
                    start_idx = i

                # 检查连续天数
                consec = (df["ET_rate"].iloc[start_idx:i+1] >= et_threshold).sum()
                if consec >= min_len:
                    in_event = True
            else:
                start_idx = None

        # ----------------------
        # 已经在事件中 → 检查结束条件
        # ----------------------
        else:
            if (et < et_threshold) or (dtheta > 0):
                end_idx = i - 1
                event_df = df.iloc[start_idx:end_idx+1].copy()

                events.append({
                    "start": df["DATE"].iloc[start_idx],
                    "end": df["DATE"].iloc[end_idx],
                    "length": end_idx - start_idx + 1,
                    "data": event_df
                })

                # 重置
                in_event = False
                start_idx = None

    return events

