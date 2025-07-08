import sys
import os
import re
import pandas as pd
import numpy as np


def get_stats_from_runtime_log(path: str, out_path: str, print_only=False):
    df = pd.read_csv(path, header=None).transpose()
    
    stats_df = pd.DataFrame(columns=["mean","std","median","max","min"])
    for col in df:
        stats_df.loc[len(stats_df)] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "median": df[col].median(),
            "max": df[col].max(),
            "min": df[col].min()
        }

    print(stats_df)
    if not print_only:
        stats_df.to_csv(os.path.join(out_path, "stats.csv"))


def parse_logs(path: str, out_path: str, print_only=False):
    gpu_util_logs = [f for f in os.listdir(path) if "gpu_util" in f]
    runtime_logs = [f for f in os.listdir(path) if "runtime" in f]

    for log in runtime_logs:
        batch_size = log.split("bsize")[1].split("_")[0]
        print(f"BATCH SIZE: {batch_size} RUNTIMES")
        get_stats_from_runtime_log(os.path.join(path, log), 
                                   os.path.join(out_path, f"bsize_{batch_size}"), 
                                   print_only=print_only)

    for log in gpu_util_logs:
        batch_size = log.split("bsize")[1].split(".")[0]
        print(f"BATCH SIZE: {batch_size} GPU UTILS")
        get_stats_from_runtime_log(os.path.join(path, log), "", print_only=True)


if __name__ == "__main__":
    log_path = sys.argv[1]
    print_only = "p" in sys.argv
    out_path = "results" if len(sys.argv) == 2 else sys.argv[2]
    
    os.makedirs(out_path, exist_ok=True)

    parse_logs(log_path, out_path, print_only)
