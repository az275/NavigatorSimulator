from queue import PriorityQueue

from core.config import *
from core.task import Task
from core.job import Job
from core.batch import Batch
from core.events import *
from core.network import *
from core.workflow import *

from workers.worker import Worker

from schedulers.centralized.shepherd.shepherd_state import ShepherdState

# import numpy as np


class OrderedTask:
    """
        Task wrapper for PriorityQueue. Ordered by increasing task arrival time.
    """

    def __init__(self, task: Task, current_time: float):
        self.task = task
        self.task_arrival_time = current_time
        self.deadline = current_time + task.slo

    def __lt__(self, other):
        return self.deadline < other.deadline
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] {self.task}"
    
    def __repr__(self):
        return self.__str__()


def _drop_bad_tasks(state: ShepherdState, model_queue: list[OrderedTask], time: float):
    skipped_tasks = []
    while model_queue.qsize() > 0:
        ot = model_queue.get()
        if time < ot.task.log.task_arrival_at_scheduler_timestamp:
            skipped_tasks.append(ot)
            continue
        # drop tasks whose SLOs can't be satisfied within a grace period
        # earliest task end time >= deadline + grace period
        if (time + ot.task.mig_batch_exec_time[24][0]) >= ot.deadline * (1 + SLO_SLACK):
            ShepherdState.task_drop_log.loc[len(ShepherdState.task_drop_log)] = {
                "job_id": ot.task.job_id,
                "workflow_id": ot.task.task_type[0],
                "task_id": ot.task.task_type[1],
                "drop_time": time,
                "arrival_time": ot.task_arrival_time,
                "slo": ot.task.slo,
                "deadline": ot.deadline
            }
        else:
            skipped_tasks.append(ot)
    
    for ot in skipped_tasks:
        model_queue.put(ot)


def _flex_form_largest_batch(state: ShepherdState, model_queue: list[OrderedTask], time: float) -> Batch:
    tasks = []
    skipped_tasks = []
    while model_queue.qsize() > 0:
        ot = model_queue.get()
        if time < ot.task.log.task_placed_on_worker_queue_timestamp:
            skipped_tasks.append(ot)
            continue
        tasks.append(ot.task)
        if len(tasks) == tasks[0].max_batch_size:
            break
    for ot in skipped_tasks:
        model_queue.put(ot)
    state.update_batch_counter()
    return Batch(ShepherdState._batch_counter-1, tasks)


def _flex_get_largest_candidate_batch(task_types: list[tuple[int,int]], 
                                      model_queues: dict[int, PriorityQueue], current_time: float):
    """
        Returns (model_id, batch_size) of the largest batch that can be formed from currently 
        queued tasks across all of [model_queues] where model_id is required by some task in
        [task_types].
    """
    largest_batch_model_id = -1
    largest_batch_size = 0
    for task_type in task_types:
        mid = get_model_id_for_task_type(task_type)
        if mid not in model_queues or model_queues[mid].qsize() == 0:
            continue # no tasks queued
        candidate_batch_size = min(len([t for t in model_queues[mid].queue if current_time >= t.task.log.task_placed_on_worker_queue_timestamp]),
                                model_queues[mid].queue[0].task.max_batch_size)
        if candidate_batch_size > largest_batch_size:
            largest_batch_size = candidate_batch_size
            largest_batch_model_id = mid
    return (largest_batch_model_id, largest_batch_size)


def flex_schedule_job_on_arrival(simulation, state: ShepherdState, model_queues: dict[int, PriorityQueue], job: Job, current_time: float):
    arrived_groups = set()
    for task in job.tasks:
        if len(task.required_task_ids) == 0:
            if task.model.model_id not in model_queues:
                model_queues[task.model.model_id] = PriorityQueue()
            model_queues[task.model.model_id].put(OrderedTask(task, current_time))
            arrived_groups.add(state.task_type_to_group[task.task_type])

    events = []
    for group in arrived_groups:
        events += flex_schedule_tasks_on_arrival(simulation, state, group, model_queues, current_time)
    return events


def flex_schedule_tasks_on_arrival(simulation, state: ShepherdState, group: int, model_queues: dict[int, PriorityQueue], 
                                   current_time: float):
    """
        While there are unchecked workers and queued tasks, creates the largest batch
        possible across all models and attempts to assign a worker to the batch in order
        of decreasing estimated execution start time.
    """
    for mq in model_queues.values():
        _drop_bad_tasks(state, mq, current_time)
    
    events = []
    
    unassigned_workers = state.worker_groups[group].copy()
    unassigned_workers = unassigned_workers[state.next_worker_idxs[group]:] + unassigned_workers[:state.next_worker_idxs[group]]

    worker_idx = 0

    largest_batch_model_id, largest_batch_size = _flex_get_largest_candidate_batch(
        state.group_task_types[group], model_queues, current_time)
    while worker_idx < len(unassigned_workers) and largest_batch_size > 0:
        next_worker = unassigned_workers[worker_idx]
        # best_worker = min(unassigned_workers,
        #                   key=lambda w: w.get_wait_time(current_time, largest_batch_model_id))
        
        if not ENABLE_DYNAMIC_MODEL_LOADING:
            if all(m.model_id != largest_batch_model_id for m in next_worker.GPU_state.placed_models(current_time)):
                worker_idx += 1
                # unassigned_workers.remove(next_worker)
                continue
        
        # when it is impossible for worker to load model for some reason
        if next_worker.get_wait_time(current_time, largest_batch_model_id) == np.inf:
            worker_idx += 1
            # unassigned_workers.remove(next_worker)
            continue

        # NOTE: workers are assumed to run only 1 batch at a time
        curr_batch = state.worker_states[next_worker.worker_id]
        curr_batch_size = curr_batch.size() if not curr_batch is None else 0

        if curr_batch_size == 0:
            # assign batch to best worker
            batch = _flex_form_largest_batch(state, model_queues[largest_batch_model_id], current_time)
            state.assign_batch_to_worker(next_worker.worker_id, batch)
            events.append(EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                BatchArrivalAtWorker(simulation, next_worker, batch)))
            # update candidate batch
            largest_batch_model_id, largest_batch_size = _flex_get_largest_candidate_batch(
                state.group_task_types[group], model_queues, current_time)
        elif largest_batch_size >= FLEX_LAMBDA * curr_batch_size:
            # assign batch to best worker
            batch = _flex_form_largest_batch(state, model_queues[largest_batch_model_id], current_time)
            old_batch_id = state.worker_states[next_worker.worker_id].id
            state.preempt_batch_on_worker(next_worker.worker_id, batch)
            events.append(EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                BatchPreemptionAtWorker(simulation, next_worker, batch, old_batch_id)))
            # update candidate batch
            largest_batch_model_id, largest_batch_size = _flex_get_largest_candidate_batch(
                state.group_task_types[group], model_queues, current_time)
        
        # remove worker from consideration
        worker_idx += 1
        # unassigned_workers.remove(best_worker)

    state.next_worker_idxs[group] = (state.next_worker_idxs[group] + worker_idx) % len(state.next_worker_idxs)
    
    return events


def flex_schedule_on_batch_completion(simulation, state: ShepherdState, model_queues: dict[int, PriorityQueue], 
                                      worker: Worker, completed_batch: Batch, current_time: float):
    for mq in model_queues.values():
        _drop_bad_tasks(state, mq, current_time)
    
    # if alr. assigned to a new batch do nothing
    if state.worker_states[worker.worker_id].id != completed_batch.id:
        return []
    
    state.worker_completed_batch(worker.worker_id, completed_batch)

    all_task_types = []
    for m in worker.GPU_state.placed_models(current_time):
        all_task_types += get_task_types_for_model(m.model_id)
    
    largest_batch_model_id, largest_batch_size = _flex_get_largest_candidate_batch(
        all_task_types if ENABLE_DYNAMIC_MODEL_LOADING else \
            state.group_task_types[state.task_type_to_group[completed_batch.tasks[0].task_type]], 
        model_queues, current_time)

    if largest_batch_size > 0:
        batch = _flex_form_largest_batch(state, model_queues[largest_batch_model_id], current_time)
        state.assign_batch_to_worker(worker.worker_id, batch)
        return [EventOrders(
            current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
            BatchArrivalAtWorker(simulation, worker, batch))]
    return []