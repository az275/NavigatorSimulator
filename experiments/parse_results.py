import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from core.workflow import *
from functools import reduce

import numpy as np


# TODO: verify units
def plot_response_time_vs_arrival_time(job_df, out_path):
    plt.figure(figsize=(10, 6))

    job_types = set(job_df["workflow_type"])
    job_names = { job_type: list(filter(lambda job: job["JOB_TYPE"]==job_type, WORKFLOW_LIST))[0]["JOB_NAME"] 
                 for job_type in job_types }

    fst_job_create_time = job_df["job_create_time"][0]
    for jt in job_types:
        job_create_times = job_df[job_df["workflow_type"] == jt]["job_create_time"] - fst_job_create_time
        job_response_times = job_df[job_df["workflow_type"] == jt]["response_time"]

        plt.scatter(
            job_create_times,
            job_response_times,
            label=f"Workflow {jt}: {job_names[jt]}",
            s=4
        )
    
    plt.xlabel("Job arrival time (ms since start)")
    plt.ylabel("Response time (ms)")
    plt.title("Response Time vs. Arrival Time by Job Type")

    plt.legend()
    plt.savefig(os.path.join(out_path, "response_vs_arrival.png"))


def plot_batch_size_vs_batch_start(batch_df, out_path):
    task_types = list(map(tuple, batch_df[['workflow_id', 'task_id']].drop_duplicates().values))

    for task_type in task_types:
        df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        
        fig = plt.figure(figsize=(10, 6))

        plt.scatter(
            df["time"],
            df["batch_size"],
            label=f"Job type {task_type[0]}, Task {task_type[1]}",
            s=4
        )
    
        plt.xlabel("Batch exec start time (ms since start)")
        plt.ylabel("Batch size")
        plt.title("Batch Size vs. Time by Model")

        plt.legend()
        plt.savefig(os.path.join(out_path, f"wf_{task_type[0]}_task_{task_type[1]}_batch_size_vs_time.png"))

        plt.close()


def plot_batch_size_bar_chart(batch_df, out_path):
    task_types = list(map(tuple, batch_df[['workflow_id', 'task_id']].drop_duplicates().values))

    for task_type in task_types:
        df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        unique_batch_sizes = set(df["batch_size"])

        fig = plt.figure(figsize=(8, 6))
        batch_size_counts = list(map(lambda size: (df["batch_size"] == size).sum(), 
                                     unique_batch_sizes))

        plt.bar(range(len(unique_batch_sizes)), batch_size_counts)
    
        plt.xticks(range(len(unique_batch_sizes)), unique_batch_sizes)
        plt.xlabel("Batch size")
        plt.ylabel("Number of batches")
        plt.title(f"Batch sizes over execution for Workflow {task_type[0]}, Task {task_type[1]}")

        plt.savefig(os.path.join(out_path, f"wf_{task_type[0]}_task_{task_type[1]}_batch_size_bar_plot.png"))
        plt.close()


def stats_by_task_type(task_df, batch_df, out_path):
    task_type_df = pd.DataFrame(columns=["workflow_id","task_id","mean_queueing_time_ms",
                                        "queueing_time_stddev","mean_batch_size","batch_size_stddev",
                                        "max_batch_size","min_batch_size","mean_exec_time_ms",
                                        "exec_time_stddev","p95_exec_time", "mean_arrival_at_worker_interval_ms",
                                        "p95_arrival_at_worker_interval_ms"])
    task_types = list(map(tuple, task_df[['workflow_type', 'task_id']].drop_duplicates().values))
    for task_type in task_types:
        task_type_task_df = task_df[(task_df["workflow_type"]==task_type[0]) & (task_df["task_id"]==task_type[1])]
        task_type_batch_df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        task_arrival_diffs = task_type_task_df.groupby("worker_id")["task_arrival_time"].apply(
            lambda x: x.diff().mean())

        task_type_df.loc[len(task_type_df)] = {
            "workflow_id": task_type[0],
            "task_id": task_type[1],
            "mean_queueing_time_ms": task_type_task_df["time_spent_in_queue"].mean(),
            "queueing_time_stddev": task_type_task_df["time_spent_in_queue"].std(),
            "mean_batch_size": task_type_batch_df["batch_size"].mean(),
            "batch_size_stddev": task_type_batch_df["batch_size"].std(),
            "max_batch_size": task_type_batch_df["batch_size"].max(),
            "min_batch_size": task_type_batch_df["batch_size"].min(),
            "mean_exec_time_ms": task_type_task_df["execution_time"].mean(),
            "exec_time_stddev": task_type_task_df["execution_time"].std(),
            "p95_exec_time": task_type_task_df["execution_time"].quantile(0.95),
            "mean_arrival_at_worker_interval_ms": task_arrival_diffs.mean(),
            # "arrival_at_worker_interval_stddev": task_arrival_diffs.std(),
            "p95_arrival_at_worker_interval_ms": np.quantile(task_arrival_diffs, 0.95) 
        }
    task_type_df.to_csv(os.path.join(out_path, "stats_by_task_type.csv"))


def plot_queueing_time_over_time(task_df, out_path):
    task_types = list(map(tuple, task_df[['workflow_type', 'task_id']].drop_duplicates().values))
    for task_type in task_types:
        task_type_task_df = task_df[(task_df["workflow_type"]==task_type[0]) & (task_df["task_id"]==task_type[1])]
        
        plt.scatter(
            task_type_task_df["task_arrival_time"],
            task_type_task_df["time_spent_in_queue"],
            s=4
        )
    
        plt.xlabel("Task arrival at worker time (ms)")
        plt.ylabel("Time spent in worker queue (ms)")
        plt.title(f"Workflow {task_type[0]}, Task {task_type[1]} Queueing Time over Time")

        plt.savefig(os.path.join(out_path, f"wf_{task_type[0]}_task_{task_type[1]}_queue.png"))
        plt.close()


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
    

def gen_stats(job_df, event_df):
    print(f"Mean response time: {job_df["response_time"].mean()}, Max: {job_df["response_time"].max()}")
    # print(f"TPUT: {len(job_df) / event_df.loc[len(events_df)-1]["time"]}")
    

def plot_model_loading_histogram(model_df, out_path):
    fig = plt.figure(figsize=(8, 6))

    plt.hist(model_df[model_df["placed_or_evicted"] == "placed"]["start_time"], bins=15, edgecolor='black')

    plt.xlabel("Time")
    plt.ylabel("Number of models loaded")
    plt.title(f"Model Loading Over Time")

    plt.savefig(os.path.join(out_path, f"model_loading_hist.png"))


def plot_model_eviction_histogram(model_df, out_path):
    fig = plt.figure(figsize=(8, 6))

    plt.hist(model_df[model_df["placed_or_evicted"] == "evicted"]["start_time"], bins=15, edgecolor='black')

    plt.xlabel("Time")
    plt.ylabel("Number of models evicted")
    plt.title(f"Model Eviction Over Time")

    plt.savefig(os.path.join(out_path, f"model_eviction_hist.png"))


results_dir_path = sys.argv[1] # results/<scheduler_type>
out_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_results"

os.makedirs(out_path, exist_ok=True)

job_df = pd.read_csv(os.path.join(results_dir_path, "job_breakdown.csv"))
task_df = pd.read_csv(os.path.join(results_dir_path, "loadDelay_1_placementDelay_1.csv"))
# events_df = pd.read_csv(os.path.join(results_dir_path, 'events_by_time.csv'))
model_df = pd.read_csv(os.path.join(results_dir_path, "model_history_log.csv"))

batch_df = pd.read_csv(os.path.join(results_dir_path, 'batch_log.csv'))

# plot_model_loading_histogram(model_df, out_path)
# plot_model_eviction_histogram(model_df, out_path)
# plot_batch_size_bar_chart(batch_df, out_path)
# plot_batch_size_vs_batch_start(batch_df, out_path)
# plot_response_time_vs_arrival_time(job_df, out_path)

stats_by_task_type(task_df, batch_df, out_path)
# plot_queueing_time_over_time(task_df, out_path)