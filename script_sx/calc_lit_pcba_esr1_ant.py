import subprocess
import tempfile
import os
import os.path as osp
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from collections import defaultdict
from tempfile import TemporaryDirectory

from script.runXGB import run_XGB

DROOT = "/vast/sx801/geometries/LIT-PCBA/ESR1_ant"

def xgb_wrapper(args):
    pro, lig = args
    with TemporaryDirectory() as temp_dir:
        # goto a temp_dir to avoid generation of temp file in the current directory
        os.chdir(temp_dir)
        res = run_XGB(pro, lig, return_linf9=True)
    if len(res) == 3:
        assert res[0] is None, res
        return None
    
    assert len(res) == 2, res
    xgb_score, linf9_score = res
    lig_id = osp.basename(lig).split(".pdb")[0]
    res = {"lig_id": lig_id, "xgb_score": xgb_score, "linf9_score": linf9_score}
    return res

def run():
    lig_pdbs = glob(osp.join(DROOT, "pose", "*.pdb"))[:10]
    prot_mol2s = glob(osp.join(DROOT, "*_protein.mol2"))
    for prot_mol2 in prot_mol2s:
        prot_pdb = prot_mol2.replace(".mol2", ".pdb")
        # a weird bug from obabel
        prot_pdb1 = prot_mol2.replace(".mol2", "1.pdb")
        if not osp.exists(prot_pdb) and not osp.exists(prot_pdb1):
            conv_cmd = f"/ext3/miniconda3/bin/obabel -imol2 {prot_mol2} -m -opdb -O {prot_pdb}"
            subprocess.run(conv_cmd, shell=True, check=True)
    
    mp_args = []
    for lig in lig_pdbs:
        pdb = osp.basename(lig).split(".pdb")[0].split("_")[-1]
        prot_pdb = osp.join(DROOT, f"{pdb}_protein1.pdb")
        mp_args.append((prot_pdb, lig))
    
    # res = [xgb_wrapper(arg) for arg in mp_args]
    res = process_map(xgb_wrapper, mp_args, max_workers=12, chunksize=10)

    res = [r for r in res if r is not None]

    res_dfs = []
    for r in res:
        this_df = pd.DataFrame(r, index=[0])
        res_dfs.append(this_df)
    del res
    res_dfs = pd.concat(res_dfs)
    res_dfs.to_csv("linf9_xgb_scores.csv", index=False)

if __name__ == "__main__":
    run()
