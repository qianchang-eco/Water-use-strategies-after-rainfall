import pandas as pd
import numpy as np

# ----------------------------
# 1. 读取数据
# ----------------------------
infile = "/data/workspace/hywang/changqian/results_q/rewet_q_results_time.csv"
df = pd.read_csv(infile)

# 只保留关心的生态系统
df = df[df["ecosystem"].isin(["grassland", "forest", "savanna"])]

# ----------------------------
# 2. 需要统计的窗口
# ----------------------------
windows = [7, 14, 21]

out_rows = []

# ----------------------------
# 3. 循环统计
# ----------------------------
for eco in ["grassland", "forest", "savanna"]:
    df_eco = df[df["ecosystem"] == eco]

    for win in windows:
        col = f"q_{win}_theta_jump"

        if col not in df_eco.columns:
            continue

        values = df_eco[col].dropna()

        if len(values) < 5:
            # 样本太少直接跳过（避免误导）
            continue

        row = {
            "ecosystem": eco,
            "window": win,
            "n_events": len(values),
            "theta_jump_median": values.median(),
            "theta_jump_mean": values.mean(),
            "theta_jump_iqr": values.quantile(0.75) - values.quantile(0.25),
            "theta_jump_q25": values.quantile(0.25),
            "theta_jump_q75": values.quantile(0.75),
        }

        out_rows.append(row)

# ----------------------------
# 4. 输出结果表
# ----------------------------
out_df = pd.DataFrame(out_rows)

outfile = "/data/workspace/hywang/changqian/results_q/theta_jump_summary_by_ecosystem.csv"
out_df.to_csv(outfile, index=False)

print("Summary table saved to:", outfile)
print(out_df)

