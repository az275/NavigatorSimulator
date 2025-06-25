from core.workflow import *
from core.model import *
from core.task import *
from core.config import *

import numpy as np


class Job(object):

    def __init__(self, create_time, job_type_id, job_id, use_boost=False):
        """
        A job is a unique object across the simulation execution that has a specific graph of task dependencies (job_type_id)
        """

        self.id = job_id  # unique ID for each job
        self.job_type_id = job_type_id
        self.job_name, self.tasks = None, []  # List of Task objects
        self.tasks = []
        # TODO: this is called everytime now. OPTIMIZE BY CALLING IT ONLY ONCE
        self.job_generate_from_workflow()
        self.ADFG = {}     # Activated Dataflow Graph scheduled by scheduler. map: task_id->worker_id
        # List containing which tasks tha constitute the job have been completed : [(task,timestamp),...]
        self.completed_tasks = []
        self.create_time = create_time  
        self.end_time = create_time
        self.use_boost = use_boost

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Job):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "JobID: {}".format(self.id)

    def get_task_by_id(self, task_id):
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    _TOTAL_JOB_TIME = 0
    _STEP_EXEC_TIME = 1
    _REMAINING_JOB_TIME = 2

    def _get_processing_time(self, available_tasks: list[Task]) -> float:
        dependencies: dict[int, set[int]] = {}
        dependents: dict[int, set[int]] = {}
        available_tasks: list[Task] = []
        for task in self.tasks:
            dependencies[task.task_id] = set(task.required_task_ids)
            dependents[task.task_id] = set(task.next_task_ids)
        
        max_cum_processing_time = 0
        while available_tasks:
            next_available_tasks = []
            # TODO: Figure out the right way to compute "processing time" here.
            # Do we include things like GPU_to_GPU_delay? Does it make sense to
            # use execution_time from a previous run to start?
            max_cum_processing_time += max([
                task.task_exec_duration for task in available_tasks
            ])
            for task in available_tasks:
                for dep in dependents[task.task_id]:
                    dependencies[dep].remove(task.task_id)
                    if len(dependencies[dep]) == 0:
                        next_available_tasks.append(self.get_task_by_id(dep))

            available_tasks = next_available_tasks
        return max_cum_processing_time

    def boost_size(self, boost_policy: int) -> float:
        """
        Compute a boost for the job based on the [boost_policy].
            boost_policy=_TOTAL_JOB_TIME returns the total processing time 
            needed to traverse the ADFG.

            boost_policy=_REMAINING_JOB_TIME returns the total remaining
            processing time.
        """
        assert(boost_policy in [self._TOTAL_JOB_TIME, self._REMAINING_JOB_TIME])

        if boost_policy == self._TOTAL_JOB_TIME:
            return self._get_processing_time([task for task in self.tasks 
                                              if len(task.required_task_ids) == 0])
        elif boost_policy == self._REMAINING_JOB_TIME:
            return self._get_processing_time([task for task in self.tasks 
                                              if task.log.task_execution_end_timestamp > 0])
    
    def assign_ADFG(self, ADFG):
        """
        Function to assign the ADFG to the job and tasks within the job
        :param ADFG: ADFG to be assigned to the job
        """
        self.ADFG = ADFG
        for task in self.tasks:
            task.ADFG = ADFG

    def job_completed(self, completion_time, task_id) -> bool:
        """ 
        Check if the all tasks in the job have completed
        Returns True if the job has completed, and False otherwise. 
        """
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        self.end_time = max(completion_time, self.end_time)
        assert len(self.completed_tasks) <= len(self.tasks)
        return len(self.completed_tasks) == len(self.tasks)

    def job_generate_from_workflow(self):
        """
        Access WORKFLOW_LIST in workflow.py and fill up the self members based on the job_type_id
        """
        job_cfg = WORKFLOW_LIST[self.job_type_id]
        self.job_name = job_cfg["JOB_NAME"]
        
        for task_cfg in job_cfg["TASKS"]:
            required_model_for_task = None
            if task_cfg["MODEL_ID"] > -1:
                required_model_for_task = Model(job_type_id=job_cfg["JOB_TYPE"],
                                                model_id=task_cfg["MODEL_ID"],
                                                model_size=task_cfg["MODEL_SIZE"])

            current_task = Task(self.id,  # ID of the associated unique Job
                                task_cfg["TASK_INDEX"],  # taskID
                                (self.job_type_id, task_cfg["TASK_INDEX"]), # task type
                                task_cfg["EXECUTION_TIME"], 
                                required_model_for_task, 
                                task_cfg["INPUT_SIZE"],
                                task_cfg["OUTPUT_SIZE"],
                                task_cfg["MAX_BATCH_SIZE"],
                                task_cfg["MAX_WAIT_TIME"],
                                task_cfg["BATCH_SIZES"],
                                task_cfg["BATCH_EXEC_TIME"],
                                task_cfg["MIG_BATCH_EXEC_TIMES"],
                                task_cfg["EXEC_TIME_COEFFICIENT_OF_VARIATION"])

            self.tasks.append(current_task)

        # Assign dependencies among Tasks
        for current_task_index in range(len(job_cfg["TASKS"])):
            for prev_idx in job_cfg["TASKS"][current_task_index]["PREV_TASK_INDEX"]:
                self.tasks[current_task_index].required_task_ids.append(prev_idx)
            for next_idx in job_cfg["TASKS"][current_task_index]["NEXT_TASK_INDEX"]:
                self.tasks[current_task_index].next_task_ids.append(next_idx)

    def assign_priorities(self, boost_parameter: float, boost_policy: int):
        """
        Assign priorities to tasks based on the ADFG
        """
        # TODO: Use more flexible priority scheme if needed in future. Default
        # to boost for now for quick testing.
        job_size = self.boost_size(boost_policy)
        for task in self.tasks:
            task.priority = self.create_time - 1 / boost_parameter * np.log(
                1 / (1 - np.exp(-boost_parameter * job_size))
            )

    def finished_task(self, task):
        for f_task in self.completed_tasks:
            if task.job_id == self.id and task.task_id == f_task[0].task_id:
                return True
        return False

    def print_job_info(self):
        for task in self.tasks:
            print("task{}, duration{}, required_task_ids{}".format(task.task_id, task.task_exec_duration,
                                                                task.required_task_ids))
