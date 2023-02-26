from collections import defaultdict
import subprocess
from script.runXGB import run_XGB

import tempfile
import os
import os.path as osp
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import os

from script_sx.casf_utils import CASF_ROOT, CORE_SET_ROOT, calc_docking_score

def xgb_wrapper(decoy, pro, return_name):
    os.chdir(osp.dirname(decoy))
    return run_XGB(pro, decoy, return_name)

def process_one_target(pdb):
    out_csv = f"/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_dock/{pdb}_score.dat"
    if osp.exists(out_csv):
        return
        
    with tempfile.TemporaryDirectory() as run_dir:
        result_dict = defaultdict(lambda: [])

        src_pdb = osp.join(CORE_SET_ROOT, pdb, f"{pdb}_protein.pdb")
        run_pdb = osp.join(run_dir, osp.basename(src_pdb))
        shutil.copyfile(src_pdb, run_pdb)
        src_lig = osp.join(CASF_ROOT, "decoys_docking", f"{pdb}_decoys.mol2")
        lig_dir = osp.join(run_dir, pdb)
        os.makedirs(lig_dir)
        tgt_tmpl = osp.join(lig_dir, f"{pdb}_decoys.pdb")
        subprocess.run(f"/ext3/miniconda3/bin/obabel -imol2 {src_lig} -m -opdb -O {tgt_tmpl}", shell=True, check=True)
        decoys = glob(osp.join(lig_dir, f"{pdb}_decoys*.pdb"))

        this_processor = partial(xgb_wrapper, pro=run_pdb, return_name=True)
        # results = process_map(this_processor, decoys, desc=pdb)
        results = [this_processor(decoy) for decoy in decoys]
        for score, name, lin_f9 in results:
            if score is None:
                continue
            result_dict["#code"].append(name)
            result_dict["score"].append(score)
            result_dict["lin_f9"].append(lin_f9)

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(out_csv, index=False, sep=" ")

def calc_dock():
    CORE_SET_ROOT = osp.join(CASF_ROOT, "coreset")
    os.makedirs("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_dock", exist_ok=True)

    pdbs = [osp.basename(f) for f in glob(osp.join(CORE_SET_ROOT, "*"))]
    process_map(process_one_target, pdbs)

def calc_dock_scores():
    linf9_save_root = "/scratch/sx801/scripts/delta_LinF9_XGB/performance/lin_f9/pred_dock"
    os.makedirs(linf9_save_root, exist_ok=True)
    xgb_linf9_save_root = "/scratch/sx801/scripts/delta_LinF9_XGB/performance/xgb_lin_f9/pred_dock"
    os.makedirs(xgb_linf9_save_root, exist_ok=True)
    scoring_df = pd.read_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_score/pred.csv").set_index("pdb")

    src_csvs = glob("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_dock/*.dat")
    for src_csv in src_csvs:
        pdb = osp.basename(src_csv).split("_")[0]
        ligand_xgb_score = scoring_df.loc[pdb]["score"]
        ligand_linf9_score = scoring_df.loc[pdb]["lin_f9"]

        this_df = pd.read_csv(src_csv, sep=" ")
        xgb_df = this_df[["#code", "score"]]
        xgb_df = pd.concat([xgb_df, pd.DataFrame({"#code": [f"{pdb}_ligand"], "score": [ligand_xgb_score]})])
        xgb_df.to_csv(osp.join(xgb_linf9_save_root, osp.basename(src_csv)), index=False, sep=" ")
        linf9_df = this_df[["#code", "lin_f9"]].rename({"lin_f9": "score"}, axis=1)
        linf9_df = pd.concat([linf9_df, pd.DataFrame({"#code": [f"{pdb}_ligand"], "score": [ligand_linf9_score]})])
        linf9_df.to_csv(osp.join(linf9_save_root, osp.basename(src_csv)), index=False, sep=" ")
    calc_docking_score(osp.dirname(xgb_linf9_save_root), "pred_dock", "docking")
    calc_docking_score(osp.dirname(linf9_save_root), "pred_dock", "docking")

if __name__ == "__main__":
    calc_dock_scores()
