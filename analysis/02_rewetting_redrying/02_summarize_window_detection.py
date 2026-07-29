import pandas as pd
import numpy as np

# 读取结果
df = pd.read_csv("/data/workspace/hywang/changqian/results_q/rewet_q_results_time.csv")

# 计算 Δq
df["dq_7"]  = df["q_7"]  - df["q_original"]
df["dq_14"] = df["q_14"] - df["q_original"]
df["dq_21"] = df["q_21"] - df["q_original"]

# 判断三列是否有值
df["has_q7"]  = ~df["q_7"].isna()
df["has_q14"] = ~df["q_14"].isna()
df["has_q21"] = ~df["q_21"].isna()

# 定义窗口唯一性
def classify_window(row):
    q7, q14, q21 = row["q_7"], row["q_14"], row["q_21"]

    # 全缺失
    if np.isnan(q7) and np.isnan(q14) and np.isnan(q21):
        return "none"

    # 只 7d
    if row["has_q7"] and not row["has_q14"] and not row["has_q21"]:
        return "only_7"

    # 只 14d
    if (not row["has_q7"]) and row["has_q14"] and (not row["has_q21"]):
        return "only_14"

    # 只 21d
    if (not row["has_q7"]) and (not row["has_q14"]) and row["has_q21"]:
        return "only_21"

    # 如果三者均有，但 q 值相同（同样峰值）
    if row["has_q7"] and row["has_q14"] and row["has_q21"]:
        if np.isclose(q7, q14) and np.isclose(q7, q21):
            return "only_7"  # 本质上返湿在 7d 内
        else:
            return "multi"

    # 多窗口情况
    return "multi"

df["window_type"] = df.apply(classify_window, axis=1)


# 分生态系统统计
ecosystems = df["ecosystem"].unique()

results = {}

for eco in ecosystems:
    sub = df[df["ecosystem"] == eco]

    total_events = len(sub)
    q7_count     = sub["has_q7"].sum()
    q14_count    = sub["has_q14"].sum()
    q21_count    = sub["has_q21"].sum()

    results[eco] = {
        "total_events": total_events,
        "q7_count": q7_count,
        "q14_count": q14_count,
        "q21_count": q21_count,
        "q7_ratio": q7_count / total_events,
        "q14_ratio": q14_count / total_events,
        "q21_ratio": q21_count / total_events,
        # 唯一窗口占比
        "only_7_ratio": (sub["window_type"]=="only_7").sum() / total_events,
        "only_14_ratio": (sub["window_type"]=="only_14").sum() / total_events,
        "only_21_ratio": (sub["window_type"]=="only_21").sum() / total_events,
        "multi_ratio": (sub["window_type"]=="multi").sum() / total_events,
    }

# 输出
results_df = pd.DataFrame(results).T
print(results_df)

# 保存
results_df.to_csv("/data/workspace/hywang/changqian/results_q/rewet_detectability_stats.csv")
print("Saved rewet_detectability_stats.csv")

