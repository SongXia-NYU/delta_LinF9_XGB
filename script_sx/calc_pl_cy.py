import json
import logging
import os
import os.path as osp
from glob import glob
import subprocess
import tempfile
import json

import pandas as pd
from rdkit.Chem import MolFromMol2Block
from tqdm.contrib.concurrent import process_map

from script.runXGB import run_XGB

TMP_FOLDER = "/vast/sx801/calc_pl_cy"
os.makedirs(TMP_FOLDER, exist_ok=True)

DATA_ROOT = "/PL_dataset/"
LIG_DATA_ROOT = "/PL_dataset/"

def _get_fl(source, pro, lig):
    ligand_fl = osp.basename(lig).split(".")[0]
    protein_fl = osp.basename(pro).split(".")[0]
    return f"{source}.{protein_fl}.{ligand_fl}"

def xgb_wrapper(arg):
    pro, lig, save_fl = arg
    with tempfile.TemporaryDirectory() as run_dir:
        try:
            os.chdir(run_dir)
            lig_cp = osp.join(run_dir, osp.basename(lig))
            pro_cp = osp.join(run_dir, osp.basename(pro))
            subprocess.run(f"cp {lig} {lig_cp}", shell=True, check=True)
            subprocess.run(f"cp {pro} {pro_cp}", shell=True, check=True)
            xgb, __, lin_f9 = run_XGB(pro_cp, lig_cp, True)
            tobesaved = xgb, save_fl, lin_f9
        except Exception as e:
            print(e)
            tobesaved = str(e), save_fl, None
    with open(osp.join(TMP_FOLDER, f"{save_fl}.json"), "w") as f:
        json.dump(tobesaved, f)


def _get_info_list(polar):
    info_list = []
    label_csvs = glob(osp.join(DATA_ROOT, "label", "*", "*.csv"))

    for csv in label_csvs:
        label_df = pd.read_csv(csv)
        split = osp.basename(osp.dirname(csv))
        source = osp.basename(csv).split(".csv")[0]

        lig_folder = "pose"
        pro_attr = ".polar" if polar else ""
        lig_attr = ".polar" if polar else ""
        prot_templ = "{pdb}_protein{pro_attr}.pdb"
        lig_templ = "{pdb}_ligand{lig_attr}.sdf"
        pdb_key = "pdb"
        if source.startswith("E2E") or source.startswith("val_E2E"):
            lig_folder = "docked_pose"
            lig_templ = "{pdb}{lig_attr}.sdf"
        elif source == "PDBbind_refined_wat":
            prot_templ = "{pdb}_protein_RW{pro_attr}.pdb"
        elif source.startswith("binder4_"):
            lig_templ = "{pdb}_docked_{o_index}{lig_attr}.sdf"
        elif source.startswith("CSAR_dry"):
            prot_templ = "{pdb}{pro_attr}.pdb"
        elif source.startswith("CSAR_decoy_"):
            prot_templ = "{pdb}{pro_attr}.pdb"
            lig_templ = "{pdb}_decoys_{o_index}{lig_attr}.sdf"
        elif source.startswith("binder5_"):
            lig_templ = "{ligand_id}_{pdb}_{o_index}{lig_attr}.sdf"
            pdb_key = "refPDB"

        for i in range(label_df.shape[0]):
            this_info = label_df.iloc[i]
            pdb = this_info[pdb_key]
            templ = {"pdb": pdb, "pro_attr": pro_attr, "lig_attr": lig_attr}
            if source.startswith("CSAR_decoy_"):
                templ["o_index"] = this_info["o_index"]
            elif source.startswith("binder4_"):
                templ["o_index"] = "{:02d}".format(this_info["o_index"])
            elif source.startswith("binder5_"):
                templ["o_index"] = "{:02d}".format(this_info["o_index"])
                templ["ligand_id"] = "{:05d}".format(this_info["ligand_id"])
            protein_pdb = osp.join(DATA_ROOT, "structure_polarH" if polar else "structure", split, source, "protein", prot_templ.format(**templ))
            ligand_sdf = osp.join(LIG_DATA_ROOT, "structure_polarH" if polar else "structure", split, source, lig_folder, lig_templ.format(**templ))
            if not osp.exists(protein_pdb):
                print(protein_pdb)
            if not osp.exists(ligand_sdf):
                print(ligand_sdf)
            
            info_list.append((protein_pdb, ligand_sdf, _get_fl(source, protein_pdb, ligand_sdf)))

    return info_list


def _json_opener(f):
    with open(f) as fp:
        return json.load(fp)

def pred_pl_dataset():

    info_list = _get_info_list(polar=True)

    print(len(info_list))
    process_map(xgb_wrapper, info_list, chunksize=10)

    results = [_json_opener(f) for f in glob(osp.join(TMP_FOLDER, "*.json"))]
    out_df = pd.DataFrame(results, columns=["xgb", "lig_name", "lin_f9"])
    out_df.to_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pl_ds.csv", index=False)

def pred_pl_second_round():
    # A second round for everyone
    info_list = _get_info_list(polar=True)
    first_round_df = pd.read_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pl_ds.csv")
    null_names = set(first_round_df[first_round_df["lin_f9"].isnull()]["lig_name"].values.tolist())
    existing_files = set([osp.basename(f).split(".json")[0] for f in glob("/vast/sx801/calc_pl_cy/*.json")])
    second_round_info = [info for info in info_list if (info[-1] in null_names) or (info[-1] not in existing_files)]
    print(len(second_round_info))

    process_map(xgb_wrapper_second_round, second_round_info, chunksize=2)

    results = [_json_opener(f) for f in glob(osp.join(TMP_FOLDER, "*.json"))]
    out_df = pd.DataFrame(results, columns=["xgb", "lig_name", "lin_f9"])
    out_df.to_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pl_ds.csv", index=False)

def xgb_wrapper_second_round(arg):
    pro, lig, save_fl = arg
    with tempfile.TemporaryDirectory() as run_dir:
        try:
            os.chdir(run_dir)
            # convert SDF files to PDB files
            lig_pdb = osp.join(run_dir, osp.basename(lig))[:-4]+'.pdb'
            pro_cp = osp.join(run_dir, osp.basename(pro))
            subprocess.run(f"/ext3/miniconda3/bin/obabel -isdf {lig} -opdb -O {lig_pdb}", shell=True, check=True)
            subprocess.run(f"cp {pro} {pro_cp}", shell=True, check=True)
            xgb, __, lin_f9 = run_XGB(pro_cp, lig_pdb, True)
            tobesaved = xgb, save_fl, lin_f9
        except Exception as e:
            print(e)
            tobesaved = str(e), save_fl, None
    with open(osp.join(TMP_FOLDER, f"{save_fl}.json"), "w") as f:
        json.dump(tobesaved, f)


if __name__ == '__main__':
    pred_pl_second_round()
