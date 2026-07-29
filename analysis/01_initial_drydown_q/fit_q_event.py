import numpy as np
from scipy.optimize import least_squares

def theta_model(t, theta0, theta_w, theta_star, ETmax, q, delta_z):
    """
    论文公式（4），q ≠ 1 情况。
    返回预测的 θ(t)
    """
    # 防止负值或无效范围
    if q == 1:
        # 线性情况（论文公式3）
        k = ETmax / (delta_z * (theta_star - theta_w))
        return (theta0 - theta_w) * np.exp(-k * t) + theta_w
    
    # 非线性情况
    A = - ETmax * (1 - q) * (theta_star - theta_w)**q / delta_z
    inner = A * t + (theta0 - theta_w)**(1 - q)
    
    inner = np.maximum(inner, 1e-12)  # 防止开方负数
    return inner**(1 / (1 - q)) + theta_w


def fit_one_event(t, theta, theta_w, theta_fc, delta_z=0.2):
    """
    输入：
        t: 事件天数数组 (0, 1, 2, ...)
        theta: 事件对应的 SWC 数组
        theta_w: 萎蔫点（站点级最小值）
        theta_fc: 场持水量（上限值）
    输出：
        拟合结果 dict（theta_star, ETmax, q, r2）
    """

    # 初值
    theta0 = theta[0]
    init_theta_star = min(theta0, theta_fc)
    init_ETmax = 2.0
    init_q = 0.8

    # 参数 bounds
    lb = [
        theta_w + 0.001,   # theta_star 下限
        0.01,              # ETmax 下限
        0.01               # q 下限
    ]
    ub = [
        theta_fc,          # theta_star 上限
        10,                # ETmax 上限 （可调整）
        3                  # q 上限
    ]

    def residuals(params):
        theta_star, ETmax, q = params
        pred = theta_model(t, theta0, theta_w, theta_star, ETmax, q, delta_z)
        return pred - theta

    # 最小二乘拟合
    try:
        res = least_squares(
            residuals,
            x0=[init_theta_star, init_ETmax, init_q],
            bounds=(lb, ub),
            max_nfev=5000,
            method='trf'
        )
    except:
        return None

    theta_star, ETmax, q = res.x
    
    # 计算 R²
    pred = theta_model(t, theta0, theta_w, theta_star, ETmax, q, delta_z)
    ss_res = np.sum((theta - pred)**2)
    ss_tot = np.sum((theta - np.mean(theta))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else -999

    return {
        "theta_star": float(theta_star),
        "ETmax": float(ETmax),
        "q": float(q),
        "r2": float(r2)
    }

