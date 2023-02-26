import os.path as osp
from glob import glob
import subprocess

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, gaussian_kde, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

CASF_ROOT = "/vast/sx801/CASF-2016-cyang"
RANKING_CSV = osp.join(CASF_ROOT, "CASF-2016.csv")
CORE_SET_ROOT = osp.join(CASF_ROOT, "coreset")

def plot_scatter_info(exp, cal, save_folder, save_name, title, total_time=None, original_size=None):
    """
    Scatter plots and scoring power
    :param exp:
    :param cal:
    :param save_folder:
    :param save_name:
    :param title:
    :param total_time:
    :param original_size:
    :return:
    """
    mae = np.mean(np.abs(exp - cal))
    rmse = np.sqrt(np.mean((exp - cal) ** 2))
    r = pearsonr(exp, cal)[0]
    r2 = r ** 2

    plt.figure()
    xy = np.vstack([exp, cal])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    plt.scatter(exp[idx], cal[idx], c=z[idx])
    x_min = min(exp)
    x_max = max(exp)
    plt.plot([x_min, x_max], [x_min, x_max], color="black", label="y==x", linestyle="--")
    plt.xlabel("Experimental")
    plt.ylabel(f"Calculated")
    plt.title(title)
    annotate = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nPearson R = {r:.4f}\n"
    if total_time is not None:
        annotate = annotate + f"Total Time = {total_time:.0f} seconds\n"
    if original_size is not None:
        annotate = annotate + f"Showing {len(exp)} / {original_size}\n"
    plt.annotate(annotate, xy=(0.05, 0.65), xycoords='axes fraction')

    # Linear regression
    lr_model = LinearRegression()
    lr_model.fit(exp.reshape(-1, 1), cal)
    slope = lr_model.coef_.item()
    intercept = lr_model.intercept_.item()
    plt.plot([x_min, x_max], [lr_model.predict(x_min.reshape(-1, 1)), lr_model.predict(x_max.reshape(-1, 1))],
             label="y=={:.2f}x+{:.2f}".format(slope, intercept), color="red")

    plt.legend()
    plt.savefig(osp.join(save_folder, save_name))
    plt.close()
    return r2


def get_rank(df, ss):
    """
    Ranking calculation, adapted from
    https://github.com/cyangNYU/delta_LinF9_XGB/blob/main/performance/Train_validation_test_performances.ipynb
    """
    spearman_list, kendall_list, target, pdb = [], [], [], []
    for i in range(1, 58):
        sa = ss.loc[ss['target'] == i]
        target.append(i)
        pdb.append(sa.pdb.tolist())
        de = df.loc[df['pdb'].isin(sa.pdb.tolist())]
        spearman_list.append(round(spearmanr(de['score'], de['pKd'])[0], 3))
        kendall_list.append(round(kendalltau(de['score'], de['pKd'])[0], 3))
    return np.mean(spearman_list), np.mean(kendall_list)

def calc_docking_score(run_dir, score_dir, out_name):
    script = osp.join(CASF_ROOT, "power_docking", "docking_power.py")
    core_set = osp.join(CASF_ROOT, "power_docking", "CoreSet.dat")
    result = osp.join(run_dir, score_dir)
    decoy = osp.join(CASF_ROOT, "decoys_docking")
    out = osp.join(run_dir, f"model_{out_name}")
    out_print = osp.join(run_dir, f"{out_name}.out")
    subprocess.run(
        f"python {script} -c {core_set} -s {result} -r {decoy} -p 'positive' -l 2 -o '{out}' > {out_print}",
        shell=True, check=True)

    result_summary = {}
    with open(out_print) as f:
        result_lines = f.readlines()
        for i, line in enumerate(result_lines):
            if line.startswith("Among the top1 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR1"] = result_lines[i+1].split()[-1]
            elif line.startswith("Among the top2 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR2"] = result_lines[i+1].split()[-1]
            elif line.startswith("Among the top3 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR3"] = result_lines[i+1].split()[-1]
    return result_summary
