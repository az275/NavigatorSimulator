import os
import argparse

import pandas as pd


def parse_batch_runtime_logs(path: str, out_path: str, drop: int, print_only=False):
    runtime_logs = [f for f in os.listdir(path) if "runtime" in f and ("bsize" in f or "batch" in f)]
    logs_with_bsizes = [(int(l.split("bsize" if "bsize" in l else "batch")[1].split("_")[0]), l) for l in runtime_logs]
    logs_with_bsizes = sorted(logs_with_bsizes, key=lambda x: x[0])

    stats_df = pd.DataFrame(columns=["bsize","median","mean","std","cv","max","min"])

    for (bsize, log) in logs_with_bsizes:
        df = pd.read_csv(os.path.join(path, log), header=None).transpose()
        df = df.iloc[drop:]
        df[0] /= 1e6

        stats_df.loc[len(stats_df)] = {
            "bsize": bsize,
            "median": df[0].median(),
            "mean": df[0].mean(),
            "std": df[0].std(),
            "cv": df[0].std() / df[0].mean(),
            "max": df[0].max(),
            "min": df[0].min()
        }

    print(f"Mean CV: {stats_df['cv'].mean()}")

    print(stats_df)
    
    if not print_only:
        stats_df.to_csv(os.path.join(out_path, "runtimes_in_ms_by_batch_size.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--print-only", action="store_true", help="Don't generate a CSV file")
    parser.add_argument("-l", "--path-to-logs", type=str, required=True, help="Path to directory where logs are stored")
    parser.add_argument("-d", "--drop", type=int, default=100, help="Num observations to drop from start")
    parser.add_argument("-o", "--out", type=str, help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    parse_batch_runtime_logs(args.path_to_logs, args.out if args.out else "", args.drop, args.print_only)
