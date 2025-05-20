import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce


def plot_response_time_vs_arrival_time(job_df, out_path):
    plt.figure(figsize=(10, 6))

    job_types = set(job_df["workflow_type"])

    fst_job_create_time = job_df["job_create_time"][0]
    for jt in job_types:
        job_create_times = job_df[job_df["workflow_type"] == jt]["job_create_time"] - fst_job_create_time
        job_response_times = job_df[job_df["workflow_type"] == jt]["response_time"]

        plt.scatter(
            job_create_times,
            job_response_times,
            label=f"type={jt}",
            s=4
        )
    
    plt.xlabel("Job arrival time")
    plt.ylabel("Response time")
    plt.title("Response Time vs. Arrival Time by Job Type")

    plt.legend()
    plt.savefig(os.path.join(out_path, "response_vs_arrival.png"))


def gen_per_task_stats(task_df, out_path):
    job_types = set(task_df["workflow_type"])
    task_types_per_job = list(map(
        lambda jt: set(task_df[task_df["workflow_type"] == jt]["task_id"]),
        job_types
    ))

    task_stat_types = ["arrival_at_worker_to_exec_start_time", "arrival_at_worker_to_enqueue_time",
                       "enqueue_to_exec_start_time", "model_fetching_time"]
    task_stats = reduce(
        lambda acc, t: acc + [f"mean_{t}", f"median_{t}", f"p99_{t}"],
        task_stat_types,
        []
    )
    task_stat_df = pd.DataFrame(columns=["job_type", "task_type"] + task_stats)

    for i, jt in enumerate(job_types):
        for task_type in task_types_per_job[i]:
            task_df_row_i = len(task_stat_df)
            task_stat_df.loc[task_df_row_i] = {"job_type": jt, "task_type": task_type}

            task_set = task_df[(task_df["workflow_type"] == jt)
                               & (task_df["task_id"] == task_type)]
            
            task_stat_data = {
                "arrival_at_worker_to_exec_start_time": task_set["task_start_exec_time"] - task_set["task_arrival_time"],
                "arrival_at_worker_to_enqueue_time": task_set["dependency_wait_time"],
                "enqueue_to_exec_start_time": task_set["time_spent_in_queue"],
                "model_fetching_time": task_set["model_fetching_time"]
            }
            for stat in task_stat_types:
                task_stat_df.loc[task_df_row_i, f"mean_{stat}"] = task_stat_data[stat].mean()
                task_stat_df.loc[task_df_row_i, f"median_{stat}"] = task_stat_data[stat].median()
                task_stat_df.loc[task_df_row_i, f"p99_{stat}"] = task_stat_data[stat].quantile(0.99)

    task_stat_df.to_csv(os.path.join(out_path, "per_task_avgs.csv"))
    

results_dir_path = sys.argv[1] # results/<scheduler_type>
out_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_results"

os.makedirs(out_path, exist_ok=True)

job_df = pd.read_csv(os.path.join(results_dir_path, "job_breakdown.csv"))
task_df = pd.read_csv(os.path.join(results_dir_path, "loadDelay_1_placementDelay_1.csv"))

plot_response_time_vs_arrival_time(job_df, out_path)
gen_per_task_stats(task_df, out_path)
