import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

from core.workflow import *
from core.config import *
from functools import reduce

import numpy as np


def plot_response_time_vs_arrival_time(job_df, out_path, plot_title_prefix):
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
    plt.title(f"{plot_title_prefix}\nResponse Time vs. Arrival Time")

    plt.legend()
    plt.savefig(os.path.join(out_path, "response_vs_arrival.pdf"))
    plt.close()


def plot_batch_size_vs_batch_start(batch_df, out_path, plot_title_prefix):
    task_types = list(map(tuple, batch_df[['workflow_id', 'task_id']].drop_duplicates().values))

    for task_type in task_types:
        df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        
        for wid in set(df["worker_id"]):
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                df[df["worker_id"]==wid]["start_time"],
                df[df["worker_id"]==wid]["batch_size"],
                s=6
            )
            plt.yticks(np.arange(2, df[df["worker_id"]==wid]["batch_size"].max()+1, 2))
            plt.xlabel("Batch exec start time (ms since start)")
            plt.ylabel("Batch size")
            plt.title(f"{plot_title_prefix}\nWorker {wid} Batch Size vs. Time for Task {task_type[1]}")
            
            plt.savefig(os.path.join(out_path, f"pipeline{task_type[0]+1}", f"task{task_type[1]}", f"worker{wid}_batch_size_vs_time.pdf"))
            plt.close()
        
        fig = plt.figure(figsize=(10, 6))

        for wid in set(df["worker_id"]):
            plt.scatter(
                df[df["worker_id"]==wid]["start_time"],
                df[df["worker_id"]==wid]["batch_size"],
                label=f"Worker {wid}",
                s=8
            )
    
        plt.yticks(np.arange(2, df["batch_size"].max()+1, 2))
        plt.xlabel("Batch exec start time (ms since start)")
        plt.ylabel("Batch size")
        plt.title(f"{plot_title_prefix}\nBatch Size vs. Time for Task {task_type[1]} By Worker")

        plt.legend()
        plt.savefig(os.path.join(out_path, f"pipeline{task_type[0]+1}", f"task{task_type[1]}", f"batch_size_vs_time_by_worker.pdf"))

        plt.close()
        

def plot_batch_size_bar_chart(batch_df, out_path, plot_title_prefix):
    task_types = list(map(tuple, batch_df[['workflow_id', 'task_id']].drop_duplicates().values))

    for task_type in task_types:
        df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        unique_batch_sizes = sorted(set(df["batch_size"]))

        fig = plt.figure(figsize=(8, 6))
        batch_size_counts = list(map(lambda size: (df["batch_size"] == size).sum(), 
                                     unique_batch_sizes))

        plt.bar(range(len(unique_batch_sizes)), batch_size_counts)
    
        plt.xticks(range(len(unique_batch_sizes)), unique_batch_sizes)
        plt.xlabel("Batch size")
        plt.ylabel("Number of batches")
        plt.title(f"{plot_title_prefix}\nBatch sizes over execution for task {task_type[1]}")

        plt.savefig(os.path.join(out_path, f"pipeline{task_type[0]+1}", f"task{task_type[1]}", f"batch_size_bar_plot.pdf"))
        plt.close()

import ast

def stats_by_task_type(task_df, batch_df, job_df, out_path):
    task_type_df = pd.DataFrame(columns=["workflow_id","task_id","mean_queueing_time_ms",
                                        "queueing_time_stddev","mean_batch_size","batch_size_stddev",
                                        "max_batch_size","min_batch_size","mean_exec_time_ms",
                                        "exec_time_stddev","p95_exec_time", "mean_arrival_at_worker_interval_ms",
                                        "p95_arrival_at_worker_interval_ms", "mean_creation_to_exec_start_ms"])
    task_types = list(map(tuple, task_df[['workflow_type', 'task_id']].drop_duplicates().values))
    for task_type in task_types:
        task_type_task_df = task_df[(task_df["workflow_type"]==task_type[0]) & (task_df["task_id"]==task_type[1])]
        task_type_batch_df = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
        task_arrival_diffs = task_type_task_df.groupby("worker_id")["task_arrival_time"].apply(
            lambda x: x.diff().mean())
            
        def _parse_data(row):
            job_ids = ast.literal_eval(row["job_ids"])
            batch_start_time = row["start_time"]
            diff_to_start = [row["start_time"] - job_df[job_df["job_id"]==jid]["job_create_time"]
                             for jid in job_ids]
            return np.mean(diff_to_start)
                       
        def _is_batch_logged(row):
            job_ids = ast.literal_eval(row["job_ids"])
            return all(jid in job_df["job_id"].values for jid in job_ids)
        
        print(task_type)
        job_creation_to_exec_start = task_type_batch_df[task_type_batch_df.apply(_is_batch_logged, axis=1)].apply(_parse_data, axis=1)

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
            "p95_arrival_at_worker_interval_ms": np.quantile(task_arrival_diffs, 0.95),
            "mean_creation_to_exec_start_ms": job_creation_to_exec_start.mean()
        }
    task_type_df.to_csv(os.path.join(out_path, "stats_by_task_type.csv"))

def plot_model_loading_histogram(model_df, out_path):
    fig = plt.figure(figsize=(8, 6))

    plt.hist(model_df[model_df["placed_or_evicted"] == "placed"]["start_time"], bins=15, edgecolor='black')

    plt.xlabel("Time")
    plt.ylabel("Number of models loaded")
    plt.title(f"Model Loading Over Time")

    plt.savefig(os.path.join(out_path, f"model_loading_hist.pdf"))
    plt.close()


def plot_model_eviction_histogram(model_df, out_path):
    fig = plt.figure(figsize=(8, 6))

    plt.hist(model_df[model_df["placed_or_evicted"] == "evicted"]["start_time"], bins=15, edgecolor='black')

    plt.xlabel("Time")
    plt.ylabel("Number of models evicted")
    plt.title(f"Model Eviction Over Time")

    plt.savefig(os.path.join(out_path, f"model_eviction_hist.pdf"))
    plt.close()

import seaborn as sns

def plot_per_task_type_latency_cdf(task_df, out_path, plot_title_prefix):
    workflows = set(task_df["workflow_type"])
    for workflow in [int(w) for w in workflows]:
        wf_df = task_df[task_df["workflow_type"]==workflow]
        task_types = set(wf_df["task_id"])
        for task_type in [int(t) for t in task_types]:
            times = wf_df[wf_df["task_id"]==task_type]["execution_time"]
        
            mean = round(np.mean(times), 2)
            median = round(np.median(times), 2)
            percentile_95 = round(np.percentile(times, 95), 2)
            variance = round(np.var(times), 2)

            sns.set_theme()
            sns.kdeplot(data=times, cumulative=True)
            plt.xlabel(f"Task execution time (ms)")
            plt.title(f"{plot_title_prefix}\nWorkflow {workflow} Task {task_type} Execution Time CDF")
            plt.annotate(f"Mean: {mean}\nMedian: {median}\nVariance: {variance}\n95th percentile: {percentile_95}",xy=(0.02, 0.8), xycoords="axes fraction", fontsize=12)
            plt.savefig(os.path.join(out_path, f"pipeline{workflow+1}", f"task{task_type}", f'latency_cdf_plot.pdf'))
            plt.close()
            

def plot_batch_sizes(batch_df, task_id, last_time, out_path, pipeline, sendrate, scheduler_name, node_count):
    fig, ax = plt.subplots(figsize=(12, 6))

    task_batch_df = batch_df[(batch_df["workflow_id"]==(pipeline-1))&(batch_df["task_id"]==task_id)]
    med_batch_sizes = [task_batch_df[(task_batch_df["start_time"] >= i*1000) & \
        (task_batch_df["start_time"]<((i+1)*1000))]["batch_size"].median()
                       for i in range(0, int(last_time / 1000), 2)]
    med_batch_sizes = [0 if np.isnan(n) else n for n in med_batch_sizes]

    ax.plot(np.arange(0, int(last_time / 1000), 2), med_batch_sizes, marker='o', markersize=3, label=f'Task {task_id}')

    ax.grid(True, axis='both', linestyle='--', alpha=0.7)

    ax.set_xlabel('Batch size')
    ax.set_ylabel('Execution start time (s since start)')
    ax.set_title(f'Pipeline {pipeline} Send Rate {sendrate} QPS {node_count}-Node {scheduler_name} Deployment\nMedian Batch Size Over Time for Task {task_id}')
    ax.legend(
        loc='upper left',
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"pipeline{pipeline}", f"task{task_id}", f'{scheduler_name}_ppl_{pipeline}_{sendrate}_task_{task_id}_batch_size_plot.pdf'))
    plt.close()

import math

def plot_latency_breakdown(stats_df, out_path, pipeline, sendrate, scheduler_name, node_count):
    latencies = {}
    # Define execution lines
    task_lines = []
    for i, row in stats_df[stats_df["workflow_id"]==(pipeline-1)].iterrows():
        task_id = int(row['task_id'])
        latencies[f"Task {task_id}"] = row["mean_exec_time_ms"]
        latencies[f"Transfer to task {task_id}"] = row["mean_creation_to_exec_start_ms"]
        prev_progress = row["mean_creation_to_exec_start_ms"] # 0 if (i == 0 or i == 1) else sum(latencies[f"Task {i}"]+latencies[f"Transfer to task {i}"] for i in range(task_id))
        task_lines.append([
            # (f"Transfer to task {task_id}", prev_progress),
            (f"Task {task_id}", prev_progress) #+ (0 if (i == 0 or i == 1) else row["mean_creation_to_exec_start_ms"]))
        ])

    # Assign distinct colors to each step
    color_map = {
        # "Transfer to task 0": "lightgray",
        # "Transfer to task 1": "lightgray",
        # "Transfer to task 2": "lightgray",
        # "Transfer to task 3": "lightgray",
        "Task 0": "red",
        "Task 1": "orange",
        "Task 2": "green",
        "Task 3": "blue"
    }

    # Y-axis level mapping
    y_levels = {
        "Transfer to task 0": 4,
        "Task 0": 4,
        "Transfer to task 1": 3,
        "Task 1": 3,
        "Transfer to task 2": 2,
        "Task 2": 2,
        "Transfer to task 3": 1,
        "Task 3": 1
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Function to draw bars
    def plot_bars_custom(line):
        for step, start_time in line:
            duration = latencies[step]
            y_level = y_levels[step]
            ax.add_patch(
                patches.Rectangle(
                    (start_time, y_level - 0.3), duration, 0.6,
                    edgecolor='black', facecolor=color_map[step], label=("Transfer" if "Transfer" in step else step)
                )
            )

    # Plot all lines
    for task_line in task_lines:
        plot_bars_custom(task_line)

    # Create legend handles (no duplicates)
    handles = []
    for step, color in color_map.items():
        handles.append(
            mlines.Line2D([], [], color=color, marker='s',
                        linestyle='None', markersize=10, label=step)
        )

    # Set axis properties
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([f"Task {int(tid)}" for tid in stats_df[stats_df["workflow_id"]==(pipeline-1)]["task_id"][::-1]])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Pipeline {pipeline} Send Rate {sendrate} QPS {node_count}-Node {scheduler_name} Deployment\nLatency breakdown for all tasks")
    
    max_x = int(max(latencies["Task 0"]+latencies["Transfer to task 0"], latencies["Task 1"]+latencies["Transfer to task 1"]) + sum(latencies[f"Transfer to task {i}"] + latencies[f"Task {i}"] for i in [2,3]))
    next_tick = math.ceil(max_x / 10**(int(math.log10(abs(max_x))))) * 10**(int(math.log10(abs(max_x))))

    ax.set_xlim(0, next_tick)
    ax.set_ylim(0.3, 4.7)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Add legend
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Component names")

    # Save and display
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"pipeline{pipeline}", "latency_breakdown.pdf"), dpi=300, bbox_inches='tight')
        

def verify_job_creation_and_arrival(job_df):
    for wid in set(job_df["workflow_type"]):
        print(f"WF {wid}")

        jobs = job_df[job_df["workflow_type"]==wid]

        for i in range(len(SEND_RATES_BY_WORKFLOW[wid]["SEND_RATES"])):
            itvl_from = sum(SEND_RATES_BY_WORKFLOW[wid]["SEND_RATE_CHANGE_INTERVALS"][:i])
            itvl_to = itvl_from + (SEND_RATES_BY_WORKFLOW[wid]["SEND_RATE_CHANGE_INTERVALS"][i] 
                if i < len(SEND_RATES_BY_WORKFLOW[wid]["SEND_RATE_CHANGE_INTERVALS"]) else len(jobs))
            itvl_jobs = jobs[max(0,itvl_from-1000):max(0,itvl_to-1000)]
            print(f"Query {itvl_from} ~ {itvl_to}")
            print(f"Mean creation interval: {itvl_jobs['job_create_time'].diff().mean()}")
            print(f"Std creation interval: {itvl_jobs['job_create_time'].diff().std()}")
            print("=================================================")
    
    # unique_workers = set(event_df["worker_id"])
    # for wid in unique_workers:
    #     if wid >= 0:
    #         arrival_events = event_df[(event_df["event"].str.contains("Job Arrival")) & (event_df["worker_id"]==wid)]
    #         print(f"WID {wid} Arrival mean: {arrival_events['time'].diff().mean()}")


if __name__ == "__main__":
    results_dir_path = sys.argv[1] # results/<scheduler_type>
    out_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_results"

    pipeline_name = sys.argv[3]
    sendrate = sys.argv[4]
    node_count = sys.argv[5]
    sched_name = sys.argv[6]

    plot_title_prefix = f"{sched_name} Pipeline {pipeline_name} Sendrate {sendrate} QPS {node_count}-Node Deployment"

    os.makedirs(out_path, exist_ok=True)

    job_df = pd.read_csv(os.path.join(results_dir_path, "job_breakdown.csv"))
    task_df = pd.read_csv(os.path.join(results_dir_path, "loadDelay_1_placementDelay_1.csv"))
    event_df = pd.read_csv(os.path.join(results_dir_path, 'events_by_time.csv'))
    batch_df = pd.read_csv(os.path.join(results_dir_path, 'batch_log.csv'))
    model_df = pd.read_csv(os.path.join(results_dir_path, "model_history_log.csv"))

    for w in set(task_df["workflow_type"]):
        for i in set(task_df[task_df["workflow_type"]==w]["task_id"]):
            os.makedirs(os.path.join(out_path, f"pipeline{int(w+1)}", f"task{int(i)}"), exist_ok=True)

    plot_model_loading_histogram(model_df, out_path)
    plot_model_eviction_histogram(model_df, out_path)

    plot_batch_size_bar_chart(batch_df, out_path, plot_title_prefix)
    plot_batch_size_vs_batch_start(batch_df, out_path, plot_title_prefix)
    plot_response_time_vs_arrival_time(job_df, out_path, plot_title_prefix)
    plot_per_task_type_latency_cdf(task_df, out_path, plot_title_prefix)
    
    stats_by_task_type(task_df, batch_df, job_df, out_path)
    # verify_job_creation_and_arrival(job_df)
    
    last_stop = round(batch_df["start_time"].max(), -3)
    
    for wf in set(batch_df["workflow_id"]):
        for task_id in set(batch_df["task_id"]):
            plot_batch_sizes(batch_df, task_id, last_stop, out_path, wf+1, sendrate, sched_name, node_count)
        # plot_latency_breakdown(pd.read_csv(os.path.join(out_path, "stats_by_task_type.csv")),
        #                        out_path, wf+1, sendrate, sched_name, node_count)
    

def get_job_stats(job_df):
    pass