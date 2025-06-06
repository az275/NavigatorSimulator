from core.logging import *
import numpy as np


class Task(object):
    def __init__(self, job_id, task_id, task_type, task_exec_duration, 
                 required_model, input_size, result_size, max_batch_size, 
                 max_wait_time, batch_sizes, batch_exec_time):
        self.job_id = job_id                           # id of the job the task belongs to
        self.task_id = task_id                         # id of the task itself
        self.task_type = task_type                     # (workflow_id, task_id)
        # the time it takes to execute the task
        self.task_exec_duration = task_exec_duration
        # required model_id to execute the task. None if it is a computation task that doesn't involve ML model
        self.model = required_model
        # task input size to model. 
        self.input_size = input_size
        self.result_size = result_size                 # output size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_sizes = batch_sizes
        self.batch_exec_time = batch_exec_time
        # list of Tasks (inputs) that this task requires ( list will be appended as the job generated)
        self.required_task_ids = []                        # list of task ids
        self.next_task_ids = []                            # list of task ids
        self.assigned_worker_id = None
        self.ADFG = {}                                  # ADFG assigned to the job that this task belongs to
        self.priority = np.inf
        self.log = TaskLifeCycleTimestamp(
            self.job_id, self.task_id)

    def __hash__(self):
        return hash((self.task_id, self.job_id))

    def __eq__(self, other):
        if (isinstance(other, Task)):
            return self.task_id == other.task_id and self.job_id == other.job_id
        return False

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __str__(self):
        return "[JobID: {}, TaskID: {} with GPU duration {}]".format(self.job_id, self.task_id, self.task_exec_duration)

    def __repr__(self):
        return self.__str__()

    def print_task_log(self):
        print(self.log.toString())
