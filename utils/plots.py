import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_off_policy_on_policy_comparison(T1, T2, T3, on_policy_vals, off_policy_vals):
    off_policy_vals = [v for v in off_policy_vals for _ in range(T3)]
    df_on_policy = pd.DataFrame(data=list(sum(on_policy_vals, [])))
    df_off_policy = pd.DataFrame(data=off_policy_vals)
    rolling_on_policy = df_on_policy.rolling(100, min_periods=1).mean()
    rolling_off_policy = df_off_policy.rolling(100, min_periods=1).mean()
    plt.plot(np.arange(0, T1 * T3), rolling_on_policy[0][:T1 * T3], label='On-policy')
    plt.plot(np.arange(0, T1 * T3), rolling_off_policy[0], label='Off-policy')
    plt.xlabel("Trajectories Count")
    plt.ylabel("Mean Return")
    plt.legend()
    plt.show()
