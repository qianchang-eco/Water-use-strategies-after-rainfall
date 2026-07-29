import pandas as pd
import numpy as np
from scipy.stats import kruskal

df = pd.read_csv("/data/workspace/hywang/changqian/results_q/rewet_q_results.csv")

#Step 1. 读取 rewet_q_results.csv，计算 Δq 
df["dq_7"]  = df["q_7"]  - df["q_original"]
df["dq_14"] = df["q_14"] - df["q_original"]
df["dq_21"] = df["q_21"] - df["q_original"]

# 只保留有效值
df_valid_7  = df[~df["dq_7"].isna()]
df_valid_14 = df[~df["dq_14"].isna()]
df_valid_21 = df[~df["dq_21"].isna()]

#Step 2. 计算生态系统 Δq 的分布统计（均值/中位数/IQR）
ecos = ["grassland", "savanna", "forest"]

stats = {}
for eco in ecos:
    sub = df[df["ecosystem"] == eco]
    stats[eco] = {
        "median_dq7":  np.nanmedian(sub["dq_7"]),
        "median_dq14": np.nanmedian(sub["dq_14"]),
        "median_dq21": np.nanmedian(sub["dq_21"]),
        "mean_dq7":    np.nanmean(sub["dq_7"]),
        "mean_dq14":   np.nanmean(sub["dq_14"]),
        "mean_dq21":   np.nanmean(sub["dq_21"]),
        "iqr_dq7":     np.nanpercentile(sub["dq_7"], 75) - np.nanpercentile(sub["dq_7"], 25),
        "iqr_dq14":    np.nanpercentile(sub["dq_14"], 75) - np.nanpercentile(sub["dq_14"], 25),
        "iqr_dq21":    np.nanpercentile(sub["dq_21"], 75) - np.nanpercentile(sub["dq_21"], 25),
    }

stats_df = pd.DataFrame(stats).T
print(stats_df)

#Step 3. 做生态系统 Δq 的显著性检验
g = df_valid_7[df_valid_7["ecosystem"]=="grassland"]["dq_7"]
s = df_valid_7[df_valid_7["ecosystem"]=="savanna"]["dq_7"]
f = df_valid_7[df_valid_7["ecosystem"]=="forest"]["dq_7"]

stat, p = kruskal(g, s, f)
print("Kruskal-Wallis for dq_7:", stat, p)

g = df_valid_14[df_valid_14["ecosystem"]=="grassland"]["dq_14"]
s = df_valid_14[df_valid_14["ecosystem"]=="savanna"]["dq_14"]
f = df_valid_14[df_valid_14["ecosystem"]=="forest"]["dq_14"]

stat, p = kruskal(g, s, f)
print("Kruskal-Wallis for dq_14:", stat, p)

g = df_valid_21[df_valid_21["ecosystem"]=="grassland"]["dq_21"]
s = df_valid_21[df_valid_21["ecosystem"]=="savanna"]["dq_21"]
f = df_valid_21[df_valid_21["ecosystem"]=="forest"]["dq_21"]

stat, p = kruskal(g, s, f)
print("Kruskal-Wallis for dq_21:", stat, p)



