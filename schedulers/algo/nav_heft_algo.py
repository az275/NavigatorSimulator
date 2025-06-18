from core.job import *
from core.task import *
from core.network import *
from core.config import *


def topological_sort(job) -> list:
    """
    Given a job DAG, compute the topological order of it. From the entry to the leaves
    """
    next_tasks_to_process = []
    sorted_ids = []
    for task in job.tasks:
        if len(task.required_task_ids) == 0:
            next_tasks_to_process.append(task)
    while(len(next_tasks_to_process) != 0):
        proc_task = next_tasks_to_process.pop()
        # check if all the preq tasks are put in sorted_list
        all_pre_proccesed = True
        for req_task_id in proc_task.required_task_ids:
            # there exist preq tasks not in sorted_list, skip this req_task
            if(req_task_id not in sorted_ids):
                all_pre_proccesed = False
                break
        if(all_pre_proccesed and proc_task.task_id not in sorted_ids):
            sorted_ids.append(proc_task.task_id)
        for next_task_id in proc_task.next_task_ids:
            if next_task_id not in sorted_ids:
                next_tasks_to_process.append(job.tasks[next_task_id])
    assert len(job.tasks) == len(sorted_ids)
    return sorted_ids


def ranking_tasks(job) -> list:
    """
    returns a sorted list of task_ids based on their upward ranking
    """
    tasks_topology = topological_sort(job)
    tasks_ranks = {}  # task_id->ranking_number
    # from the leaves to the entry
    for i in range(len(tasks_topology) - 1, -1, -1):
        cur_task_indx = tasks_topology[i]
        cur_task = job.tasks[cur_task_indx]
        cur_rank = cur_task.task_exec_duration
        max_succ_acc_ranks = 0
        # loop through the successsors
        transfer_delay = GPU_to_GPU_delay(cur_task.result_size)
        for succ_task_id in cur_task.next_task_ids:
            temp_rank = transfer_delay + tasks_ranks[succ_task_id]
            if temp_rank > max_succ_acc_ranks:
                max_succ_acc_ranks = temp_rank
        cur_rank += max_succ_acc_ranks
        tasks_ranks[cur_task.task_id] = cur_rank
    # sorted tuple list, highest rank -> lowest rank
    sorted_tasks = sorted(tasks_ranks.items(),
                          key=lambda x: x[1], reverse=True)
    sorted_task_ids = [item[0] for item in sorted_tasks]
    return sorted_task_ids


def nav_heft_job_plan(job, worker_list, current_time, initial_worker_id=None, consider_load=True, consider_cache=True) -> dict:
    """
    The planning phase scheduler is adopt and customized based on HEFT algorithm
    @Param
        worker_list: a list of worker objects
        initial_worker_id: used by decentralized scheduler
            if not None: then current_worker_id is the scheduler server that is computing the scheduling
        consider_load:
            if True: [A*] consider the machine waittime when selecting machines
            if False: [sub-optimal] do not consider current machine waittime when selecting machines, treat as if all machines are idol
        consider_cache:
            if True: [A*] consider the model transfering time when selecting machines(this method will prefer the selection of machines with model cached)
            if False: [sub-optimal] do not count into the model transfering time when selecting machines (cause more data transfering then needed)
    @Return
        dict: {task_id -> worker_id,}
    @Complexity:
        O(T^2*P)
    """
    # decided activation graph that accumulates along this function
    allocated_tasks_info = {}  # task_id -> (worker_id, finish_time)
    workers = {}
    for worker in worker_list:
        workers[worker.worker_id] = worker
    sorted_tasks = ranking_tasks(job)
    workers_to_select = [w.worker_id for w in worker_list]
    workers_EAT = {}   # worker_id -> (task_type -> earliest_available_time)
    workers_available_memory = {}  # worker_id -> available_memory
    # 1. initialize the earliest available time and memory for each worker
    for worker_id in workers_to_select:
        workers_EAT[worker_id] = {
            task_id: current_time + (workers[worker_id].get_task_queue_waittime(
                current_time,
                (job.job_type_id, task_id),
                job.tasks[task_id].model.model_size if job.tasks[task_id].model else 0,
                info_staleness=LOAD_INFORMATION_STALENESS,
                requiring_worker_id=initial_worker_id) if consider_load else 0)
            for task_id in sorted_tasks
        }
        
        available_memory = GPU_MEMORY_SIZE
        if consider_cache:
            available_memory = workers[worker_id].used_GPUmemory(current_time, \
                                                                 info_staleness=PLACEMENT_INFORMATION_STALENESS, \
                                                                 requiring_worker_id=initial_worker_id)
        workers_available_memory[worker_id] = available_memory
        
    # Select the best worker for each task based on their ranking from high to low
    for task_id in sorted_tasks:
        cur_task = job.tasks[task_id]
        # 2. among the workers to select from, calculate their earliest start time
        selected_worker_id = None
        earliest_start_time = float('inf')
        fetching_model_size = 0
        for cur_worker_id in workers_to_select:
            # 2.0 consider the current worker queue wait time to determine its earliest start time
            cur_earliest_start_time = workers_EAT[cur_worker_id][task_id]
            # 2.1 calculate the inputs arrival time
            inputs_arrival_time = 0
            if cur_task.task_id == 0 and initial_worker_id is not None and cur_worker_id != initial_worker_id:
                inputs_arrival_time += CPU_to_CPU_delay(cur_task.input_size)
            for pre_task_id in cur_task.required_task_ids:
                pre_worker = allocated_tasks_info[pre_task_id][0]
                pre_task_arrival_time = allocated_tasks_info[pre_task_id][1]
                if pre_worker != cur_worker_id:
                    pre_task_arrival_time += GPU_to_GPU_delay(job.get_task_by_id(pre_task_id).result_size)
                    inputs_arrival_time = max(pre_task_arrival_time, inputs_arrival_time)
            cur_earliest_start_time = max(cur_earliest_start_time, inputs_arrival_time)
            # 2.2 calculate the model fetch time
            model_fetch_time = 0
            cur_fetching_model_size = 0
            if consider_cache:
                models_in_cur_worker = workers[cur_worker_id].get_model_history(current_time, \
                                                                             info_staleness=PLACEMENT_INFORMATION_STALENESS, \
                                                                             requiring_workerid= initial_worker_id)
                if cur_task.model is not None and cur_task.model not in models_in_cur_worker:
                    model_fetch_time = SameMachineCPUtoGPU_delay(cur_task.model.model_size)
                    cur_fetching_model_size = cur_task.model.model_size
                    if workers_available_memory[cur_worker_id] + cur_task.model.model_size > GPU_MEMORY_SIZE:
                        # double model fetch time due to the overhead from model_eviction
                        model_fetch_time += model_fetch_time
            cur_earliest_start_time += model_fetch_time
            # 2.3 replace the selected_worker if cur_worker starts earlier
            if cur_earliest_start_time < earliest_start_time:
                earliest_start_time = cur_earliest_start_time
                selected_worker_id = cur_worker_id
                fetching_model_size = cur_fetching_model_size
        # 3. pick the worker with ealiest start time
        cur_task_finish_time = earliest_start_time + job.tasks[task_id].task_exec_duration
        workers_EAT[selected_worker_id][task_id] = cur_task_finish_time
        allocated_tasks_info[task_id] = (selected_worker_id,  cur_task_finish_time)
        if workers_available_memory[selected_worker_id] >= fetching_model_size:
            workers_available_memory[selected_worker_id] -= fetching_model_size
    allocated_tasks_worker_map = {}
    for task_id, info in allocated_tasks_info.items():
        allocated_tasks_worker_map[task_id] = info[0]
    return allocated_tasks_worker_map


def nav_heft_task_adjustment(job, task_id, workers, current_time, local_worker_id, allocated_worker_id) -> int:
    # 1. check assigned worker wait_time to decide if need to adjust assigned worker
    cur_wait_time = workers[allocated_worker_id].get_task_queue_waittime(current_time, \
                                                                        (job.job_type_id, task_id), \
                                                                        info_staleness=LOAD_INFORMATION_STALENESS, \
                                                                        requiring_worker_id=local_worker_id)
    cur_task = job.tasks[task_id]
    tolerable_wait_time = cur_wait_time <= cur_task.task_exec_duration * RESCHEDULE_THREASHOLD
    joint_task = len(cur_task.required_task_ids) > 1 
    if joint_task or tolerable_wait_time:
        return allocated_worker_id
    # 2. check if there is a better worker to assign to
    selected_worker_id = None
    earliest_start_time = float('inf')
    for cur_worker in workers:
        wait_time = cur_worker.get_task_queue_waittime(current_time, \
                                                       (job.job_type_id, task_id), \
                                                       info_staleness=LOAD_INFORMATION_STALENESS, \
                                                       requiring_worker_id=local_worker_id)
        cur_earliest_start_time = current_time + wait_time
        # 2.1 calculate the intermediate transfer time
        if  local_worker_id != cur_worker.worker_id:
            cur_earliest_start_time += CPU_to_CPU_delay(cur_task.input_size)
        # 2.2 calculate the model fetch time
        model_fetch_time = 0
        models_in_cur_worker = cur_worker.get_model_history(current_time, \
                                                            info_staleness=PLACEMENT_INFORMATION_STALENESS, \
                                                            requiring_workerid= local_worker_id)
        if cur_task.model is not None and cur_task.model not in models_in_cur_worker:
            model_fetch_time = SameMachineCPUtoGPU_delay(cur_task.model.model_size)
            available_memory = cur_worker.used_GPUmemory(current_time, \
                                                         info_staleness=PLACEMENT_INFORMATION_STALENESS, \
                                                         requiring_worker_id=local_worker_id)
            if available_memory + cur_task.model.model_size > GPU_MEMORY_SIZE:
                # double model fetch time due to the overhead from model_eviction
                model_fetch_time = model_fetch_time * 2
        cur_earliest_start_time += model_fetch_time
        # 2.3 replace the selected_worker if cur_worker starts earlier
        if cur_earliest_start_time < earliest_start_time:
            earliest_start_time = cur_earliest_start_time
            selected_worker_id = cur_worker.worker_id
    return selected_worker_id
