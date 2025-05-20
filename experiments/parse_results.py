import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from core.workflow import *
from functools import reduce


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


def plot_batch_size_vs_batch_start(event_df, out_path):
    batch_start_events = event_df[event_df["event"].str.contains("Batch Start")]

    task_types = set(batch_start_events["event"].str.extract(r"Task \(([0-9]+, [0-9]+)\)")[0])
    model_names = { 
        task_type: list(filter(
            lambda task: task["TASK_INDEX"]==int(task_type.split(", ")[1]),
            list(filter(lambda job: job["JOB_TYPE"]==int(task_type.split(",")[0]), WORKFLOW_LIST))[0]["TASKS"]
        ))[0]["MODEL_NAME"] for task_type in task_types }

    for task_type in task_types:
        type_details = [int(item) for item in task_type.split(", ")] # [workflow_id, task_id]

        fig = plt.figure(figsize=(10, 6))

        batch_start_events_for_type = batch_start_events[batch_start_events["event"].str.contains(f"Task \({task_type}\)")]
        batch_sizes = batch_start_events_for_type["event"].str.extract(r"Jobs ([0-9|,]+)")[0].str.count(f'[0-9]+')
        
        plt.scatter(
            batch_start_events_for_type["time"],
            batch_sizes,
            label=f"Workflow {type_details[0]}, Task ID {type_details[1]}: Model {model_names[task_type]}",
            s=4
        )
    
        plt.xlabel("Batch exec start time (ms since start)")
        plt.ylabel("Batch size")
        plt.title("Batch Size vs. Time by Model")

        plt.legend()
        plt.savefig(os.path.join(out_path, f"wf_{type_details[0]}_task_{type_details[1]}_batch_size_vs_time.png"))


def plot_batch_size_bar_chart(event_df, out_path):
    batch_start_events = event_df[event_df["event"].str.contains("Batch Start")]

    task_types = set(batch_start_events["event"].str.extract(r"Task \(([0-9]+, [0-9]+)\)")[0])
    task_details = { 
        task_type: list(filter(
            lambda task: task["TASK_INDEX"]==int(task_type.split(", ")[1]),
            list(filter(lambda job: job["JOB_TYPE"]==int(task_type.split(",")[0]), WORKFLOW_LIST))[0]["TASKS"]
        ))[0] for task_type in task_types }

    for task_type in task_types:
        type_details = [int(item) for item in task_type.split(", ")] # [workflow_id, task_id]

        fig = plt.figure(figsize=(8, 6))

        batch_start_events_for_type = batch_start_events[batch_start_events["event"].str.contains(f"Task \({task_type}\)")]
        batch_size_events = batch_start_events_for_type["event"].str.extract(r"Jobs ([0-9|,]+)")[0].str.count(f'[0-9]+')
        batch_size_counts = list(map(lambda size: (batch_size_events == size).sum(),
                                task_details[task_type]["BATCH_SIZES"]))

        plt.bar(
            range(len(task_details[task_type]["BATCH_SIZES"])),
            batch_size_counts
        )
    
        plt.xticks(range(len(task_details[task_type]["BATCH_SIZES"])), task_details[task_type]["BATCH_SIZES"])
        plt.xlabel("Batch sizes")
        plt.ylabel("Number of batches")
        plt.title(f"Batch size distribution for {task_details[task_type]["MODEL_NAME"]} Model")

        plt.savefig(os.path.join(out_path, f"wf_{type_details[0]}_task_{type_details[1]}_batch_size_dist.png"))


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
    

results_dir_path = sys.argv[1] # results/<scheduler_type>
out_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_results"

os.makedirs(out_path, exist_ok=True)

job_df = pd.read_csv(os.path.join(results_dir_path, "job_breakdown.csv"))
# task_df = pd.read_csv(os.path.join(results_dir_path, "loadDelay_1_placementDelay_1.csv"))
events_df = pd.read_csv(os.path.join(results_dir_path, 'events_by_time.csv'))

plot_batch_size_bar_chart(events_df, out_path)
plot_batch_size_vs_batch_start(events_df, out_path)
plot_response_time_vs_arrival_time(job_df, out_path)
