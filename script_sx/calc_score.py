from collections import defaultdict
from script.runXGB import run_XGB

import tempfile
import os
import os.path as osp
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm

from script_sx.casf_utils import plot_scatter_info

CASF_ROOT = "/vast/sx801/CASF-2016-cyang"
RANKING_CSV = osp.join(CASF_ROOT, "CASF-2016.csv")

def calc_score():
    CORE_SET_ROOT = osp.join(CASF_ROOT, "coreset_polarH")
    result_dict = defaultdict(lambda: [])

    pdbs = [osp.basename(f) for f in glob(osp.join(CORE_SET_ROOT, "*"))]
    with tempfile.TemporaryDirectory() as run_dir:
        for pdb in tqdm(pdbs[:10]):
            src_pdb = osp.join(CORE_SET_ROOT, pdb, f"{pdb}_protein.polar.pdb")
            run_pdb = osp.join(run_dir, osp.basename(src_pdb))
            shutil.copyfile(src_pdb, run_pdb)
            src_lig = osp.join(CORE_SET_ROOT, pdb, f"{pdb}_ligand.polar.mol2")
            run_lig = osp.join(run_dir, osp.basename(src_lig))
            shutil.copyfile(src_lig, run_lig)

            try:
                this_score, __, lin_f9 = run_XGB(run_pdb, run_lig, return_name=True)
            except Exception as e:
                print(src_lig)
                raise e
            result_dict["pdb"].append(pdb)
            result_dict["score"].append(this_score)
            result_dict["lin_f9"].append(lin_f9)

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_score/pred_polarH.csv", index=False)

def calc_score_power():
    linf9_save_root = "/scratch/sx801/scripts/delta_LinF9_XGB/performance/lin_f9"
    os.makedirs(linf9_save_root, exist_ok=True)
    xgb_linf9_save_root = "/scratch/sx801/scripts/delta_LinF9_XGB/performance/xgb_lin_f9"
    os.makedirs(xgb_linf9_save_root, exist_ok=True)

    scoring_df = pd.read_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_score/pred.csv").set_index("pdb")
    tgt_df = pd.read_csv(RANKING_CSV)[["pdb", "pKd"]].set_index("pdb")
    scoring_df = scoring_df.join(tgt_df)

    exp = scoring_df["pKd"].values
    cal = scoring_df["score"].values
    r2 = plot_scatter_info(exp, cal, xgb_linf9_save_root, "exp_vs_cal.png", "Experimental pKd vs. Calculated pKd")

    exp = scoring_df["pKd"].values
    cal = scoring_df["lin_f9"].values
    r2 = plot_scatter_info(exp, cal, linf9_save_root, "exp_vs_cal.png", "Experimental pKd vs. Calculated pKd")

if __name__ == "__main__":
    calc_score()
