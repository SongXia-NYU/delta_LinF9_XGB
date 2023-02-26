import pandas as pd

def analysis():
    res_df = pd.read_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pl_ds.csv")
    print(res_df.shape)
    res_df_num = res_df.dropna()
    print(res_df_num.shape)


if __name__ == '__main__':
    analysis()